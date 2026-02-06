import time
import io
import numpy as np
import supervision as sv
import ollama
import cv2
from ollama import ResponseError
from PIL import Image
from rfdetr import RFDETRNano

start = time.time()

MODEL_NAME = "qwen2.5vl:3b-q4_K_M"
IMAGE_PATH = "data/drive-download-20260205T175615Z-1-001/Копия_Аттракционы_Регистрация_Page_1_drawio.png"
WEIGHTS_PATH = "data/checkpoint_best_total_40.pth"
THRESHOLD = 0.30

MIN_SIDE = 28
UPSCALE_SMALL = True


def _pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def is_content_valid_inverse(pil_crop):
    img_np = np.array(pil_crop.convert("L"))
    h, w = img_np.shape

    if w * h < 100:
        return False, "Too small"

    _, binary = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    non_zero = cv2.countNonZero(binary)
    density = non_zero / (w * h)

    if density < 0.005:
        return False, f"Empty/Blank (Density: {density:.3f})"

    if density > 0.60:
        return False, f"Solid Fill/Blackout (Density: {density:.3f})"

    return True, "OK"


def order_actions(results, direction: str):
    if not results:
        return results

    if direction == "lr":
        return sorted(results, key=lambda r: (r["box"][0], r["box"][1]))
    if direction == "td":
        return sorted(results, key=lambda r: (r["box"][1], r["box"][0]))



def show_annotated_objects(image: Image.Image, detections: sv.Detections, class_mapping: dict):
    try:
        print("\n[VISUALIZATION] Generating annotated image...")
        image_np = np.array(image)
        box_annotator = sv.BoxAnnotator()

        labels = []
        for class_id in detections.class_id:
            class_name = class_mapping.get(class_id, str(class_id))
            labels.append(f"{class_name}")

        label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

        annotated_frame = box_annotator.annotate(scene=image_np.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        try:
            sv.plot_image(annotated_frame)
        except Exception:
            pass


    except Exception as e:
        print(f"[ERROR] Visualization failed: {e}")


def qwen_ocr_from_boxes(image, detections, class_mapping, model_name=MODEL_NAME):
    image_np = np.array(image)
    results = []

    prompt = "Extract text from image. Output text only."

    for i, (box, class_id) in enumerate(zip(detections.xyxy, detections.class_id)):
        x1, y1, x2, y2 = map(int, box)
        class_name = class_mapping.get(class_id, f"Unknown-{class_id}")

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_np.shape[1], x2), min(image_np.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = image_np[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        h, w = crop.shape[:2]
        crop_pil = Image.fromarray(crop)

        is_valid, reason = is_content_valid_inverse(crop_pil)
        if not is_valid:
            print(f"[FILTER SKIP] Class: {class_name} | Reason: {reason} | Box: {x1, y1}")
            continue


        if h < MIN_SIDE or w < MIN_SIDE:
            if not UPSCALE_SMALL:
                continue
            scale = max(MIN_SIDE / w, MIN_SIDE / h)
            new_w = max(MIN_SIDE, int(round(w * scale)))
            new_h = max(MIN_SIDE, int(round(h * scale)))
            crop_pil = crop_pil.resize((new_w, new_h), Image.BICUBIC)

        img_bytes = _pil_to_png_bytes(crop_pil)

        try:
            resp = ollama.chat(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [img_bytes],
                }],
                options={"temperature": 0.0},
            )
        except ResponseError as e:
            print(f"Ollama ResponseError for box {x1, y1, x2, y2}: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error for box {x1, y1, x2, y2}: {e}")
            continue

        text = (resp.get("message", {}) or {}).get("content", "").strip()

        results.append({
            "text": text,
            "class": class_name,
            "class_id": int(class_id),
            "box": [x1, y1, x2, y2],
            "role": None
        })

        print(f"OCR processed object {i + 1}/{len(detections.xyxy)}")

    return results


def process_tasks_with_roles(results, direction):
    LANE_CLASS_ID = 1

    lanes = [r for r in results if r["class_id"] == LANE_CLASS_ID]
    tasks = [r for r in results if r["class_id"] != LANE_CLASS_ID]

    if not lanes:
        return tasks

    for task in tasks:
        tx1, ty1, tx2, ty2 = task["box"]
        candidate_lanes = []

        for lane in lanes:
            lx1, ly1, lx2, ly2 = lane["box"]
            is_inside = False
            if direction == "lr":
                t_center_y = (ty1 + ty2) / 2
                if ly1 <= t_center_y <= ly2:
                    is_inside = True
            else:
                t_center_x = (tx1 + tx2) / 2
                if lx1 <= t_center_x <= lx2:
                    is_inside = True

            if is_inside:
                candidate_lanes.append(lane)

        if candidate_lanes:
            if len(candidate_lanes) > 1:
                if direction == "lr":
                    candidate_lanes.sort(key=lambda l: (l["box"][3] - l["box"][1]))
                else:
                    candidate_lanes.sort(key=lambda l: (l["box"][2] - l["box"][0]))

            lane_text = candidate_lanes[0]["text"].strip()
            task["role"] = lane_text if lane_text else f"Lane-ID-{candidate_lanes[0]['class_id']}"

    return tasks


print("Loading RFDETR model...")
model = RFDETRNano(pretrain_weights=WEIGHTS_PATH)
model.optimize_for_inference()

if hasattr(model, "id2label"):
    class_mapping = model.id2label
elif hasattr(model, "classes"):
    class_mapping = {i: name for i, name in enumerate(model.classes)}
else:
    class_mapping = {}

image = Image.open(IMAGE_PATH).convert("RGB")
width, height = image.size

direction = "lr" if width > height else "td"
print(f"Processing image {width}x{height}. Assumed direction: {direction.upper()}")

detections = model.predict(image, threshold=THRESHOLD)
print(f"Detections found: {len(detections.xyxy)}")

if len(detections.xyxy) > 0:
    show_annotated_objects(image, detections, class_mapping)

all_results = qwen_ocr_from_boxes(image, detections, class_mapping, model_name=MODEL_NAME)

print("\nall detected: ")
for r in all_results:
    print(f"ID: {r['class_id']} | Text: '{r['text']}' | Box: {r['box']}")

final_tasks = process_tasks_with_roles(all_results, direction)
ordered_tasks = order_actions(final_tasks, direction=direction)

print(f"\n result: ")
for i, r in enumerate(ordered_tasks, 1):
    role_info = f" | Role: {r['role']}" if r.get('role') else ""
    print(f"{i}. [Class ID: {r['class_id']}] {r['text']}{role_info}")

end = time.time()
print(f"\nExecution time: {end - start:.6f} seconds")
