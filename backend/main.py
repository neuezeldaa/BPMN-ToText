from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Tuple
import time
import io
import numpy as np
import ollama
from ollama import ResponseError
from PIL import Image
from rfdetr import RFDETRNano

app = FastAPI(version="1.0.0")

# --- КОНФИГУРАЦИЯ ---
MODEL_NAME = "qwen2.5vl:3b-q4_K_M"
WEIGHTS_PATH = "data/checkpoint_best_total_40.pth"
THRESHOLD = 0.35
MIN_SIDE = 28
UPSCALE_SMALL = True


class TextBox(BaseModel):
    text: str
    label: str
    box: List[int]
    role: Optional[str] = None
    class_id: int


class PredictResponse(BaseModel):
    direction: str
    image_size: dict
    execution_time: float
    results: List[TextBox]


print("Loading RFDETR model...")
try:
    detector = RFDETRNano(pretrain_weights=WEIGHTS_PATH)
    detector.optimize_for_inference()
    print(f"RFDETR model {WEIGHTS_PATH} with threshold = {THRESHOLD} loaded.")
except Exception as e:
    print(f"Error loading model: {e}")
    detector = None


def _pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def identify_lanes_by_position_and_ratio(detections, img_w, img_h, direction) -> List[int]:
    if len(detections.class_id) == 0:
        return []

    if direction != "lr":
        print(f"[AUTO-LANE] Skipping lane detection for direction '{direction}'. Lanes are only supported in LR.")
        return []

    boxes = detections.xyxy
    lane_indices = []

    left_edge_threshold = img_w * 0.5

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1

        if x1 < left_edge_threshold:
            if h >= (2 * w):
                lane_indices.append(i)
                print(f"[AUTO-LANE] Found vertical header at {box} (h={h:.0f}, w={w:.0f})")

    return lane_indices


def qwen_ocr_from_boxes(image, detections, model_name=MODEL_NAME):
    image_np = np.array(image)
    results = []
    prompt = "Extract text from image. Output text only."

    for i, (box, class_id) in enumerate(zip(detections.xyxy, detections.class_id)):
        x1, y1, x2, y2 = map(int, box)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_np.shape[1], x2), min(image_np.shape[0], y2)
        if x2 <= x1 or y2 <= y1: continue

        crop = image_np[y1:y2, x1:x2]
        if crop.size == 0: continue

        h, w = crop.shape[:2]
        crop_pil = Image.fromarray(crop)
        if h < MIN_SIDE or w < MIN_SIDE:
            if UPSCALE_SMALL:
                scale = max(MIN_SIDE / w, MIN_SIDE / h)
                new_w = max(MIN_SIDE, int(round(w * scale)))
                new_h = max(MIN_SIDE, int(round(h * scale)))
                crop_pil = crop_pil.resize((new_w, new_h), Image.BICUBIC)

        img_bytes = _pil_to_png_bytes(crop_pil)

        try:
            resp = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt, "images": [img_bytes]}],
                options={"temperature": 0.0},
            )
            text = (resp.get("message", {}) or {}).get("content", "").strip()

            results.append({
                "index": i,
                "text": text,
                "class_id": int(class_id),
                "box": [x1, y1, x2, y2],
                "role": None,
                "label": "Unknown"
            })
        except Exception:
            continue

    return results


def process_roles_with_indices(results, direction, lane_indices, img_w, img_h):
    lanes = []
    tasks = []

    for r in results:
        if r["index"] in lane_indices:
            r["label"] = "Lane"
            if not r["text"]: r["text"] = "Unknown Role"
            lanes.append(r)
        else:
            r["label"] = "Task"
            tasks.append(r)

    if not lanes:
        return tasks

    print(f"[INFO] Processing roles using {len(lanes)} identified lanes...")

    for task in tasks:
        tx1, ty1, tx2, ty2 = task["box"]
        tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2

        candidate_lanes = []

        for lane in lanes:
            lx1, ly1, lx2, ly2 = lane["box"]
            is_in_row = False

            if direction == "lr":
                if ly1 <= tcy <= ly2:
                    is_in_row = True
            else:
                if lx1 <= tcx <= lx2:
                    is_in_row = True

            if is_in_row:
                candidate_lanes.append(lane)

        if candidate_lanes:
            if direction == "lr":
                candidate_lanes.sort(key=lambda l: (l["box"][3] - l["box"][1]))
            else:
                candidate_lanes.sort(key=lambda l: (l["box"][2] - l["box"][0]))

            role_text = candidate_lanes[0]["text"].replace('\n', ' ').strip()
            task["role"] = role_text

    return tasks


def order_actions(results, direction: str):
    if not results: return results
    if direction == "lr":
        return sorted(results, key=lambda r: (r["box"][0], r["box"][1]))
    return sorted(results, key=lambda r: (r["box"][1], r["box"][0]))


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if detector is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    start_t = time.time()
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        w, h = image.size

        if w == 0 or h == 0:
            raise HTTPException(status_code=400, detail="Invalid image")

        direction = "LR" if w > h else "TD"
        print(f"Image: {w}x{h}, Direction: {direction}")

        detections = detector.predict(image, threshold=THRESHOLD)

        lane_indices = identify_lanes_by_position_and_ratio(detections, w, h, direction.lower())

        raw_results = qwen_ocr_from_boxes(image, detections, model_name=MODEL_NAME)

        processed_tasks = process_roles_with_indices(raw_results, direction.lower(), lane_indices, w, h)

        ordered = order_actions(processed_tasks, direction.lower())

        execution_time = time.time() - start_t

        return PredictResponse(
            direction=direction,
            image_size={"width": w, "height": h},
            execution_time=execution_time,
            results=[
                TextBox(
                    text=r["text"],
                    label=r["label"],
                    class_id=r["class_id"],
                    box=r["box"],
                    role=r.get("role")
                ) for r in ordered
            ]
        )

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "detector": "RFDETR-Nano",
        "ocr_model": MODEL_NAME
    }
