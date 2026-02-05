import time
import io
import numpy as np
import supervision as sv
import ollama
from ollama import ResponseError
from PIL import Image
from rfdetr import RFDETRNano

start = time.time()

MODEL_NAME = "qwen2.5vl:3b-q4_K_M"
IMAGE_PATH = "test/3.png"
WEIGHTS_PATH = "data/checkpoint_best_total_40.pth"
THRESHOLD = 0.5

MIN_SIDE = 28
UPSCALE_SMALL = True


def _pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def order_actions(results, direction: str):
    if not results:
        return results

    if direction == "lr":
        return sorted(results, key=lambda r: (r["box"][0], r["box"][1]))
    if direction == "td":
        return sorted(results, key=lambda r: (r["box"][1], r["box"][0]))

    raise ValueError("direction must be 'lr' or 'td'")


def qwen_ocr_from_boxes(image, detections, class_mapping, model_name=MODEL_NAME):
    image_np = np.array(image)
    results = []

    prompt = "Extract text from image. Output text only."

    for box, class_id in zip(detections.xyxy, detections.class_id):
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

        # Сохраняем результат, даже если текст пустой (чтобы видеть детекцию)
        results.append({
            "text": text,
            "class": class_name,
            "class_id": int(class_id),
            "box": [x1, y1, x2, y2],
            "role": None
        })

    return results


def process_tasks_with_roles(results, direction):
    # --- НАСТРОЙКА ID КЛАССОВ ---
    # Поменяйте эти цифры, если DEBUG покажет другие ID
    LANE_CLASS_ID = 1
    TASK_CLASS_ID = 2

    # 1. Разделяем найденные объекты
    lanes = [r for r in results if r["class_id"] == LANE_CLASS_ID]

    # Считаем "задачами" всё, что НЕ лейны (чтобы не потерять Events, Gateways)
    tasks = [r for r in results if r["class_id"] != LANE_CLASS_ID]

    # Если лэйнов нет, возвращаем просто список задач, чтобы не было пустого экрана
    if not lanes:
        print("\n[INFO] Lanes (Class ID 1) not found. Skipping role assignment.")
        return tasks

    for task in tasks:
        tx1, ty1, tx2, ty2 = task["box"]
        candidate_lanes = []

        for lane in lanes:
            lx1, ly1, lx2, ly2 = lane["box"]

            is_inside = False
            if direction == "lr":
                # Для горизонтальной: проверяем попадание центра объекта по Y
                t_center_y = (ty1 + ty2) / 2
                if ly1 <= t_center_y <= ly2:
                    is_inside = True
            else:  # td
                # Для вертикальной: проверяем попадание центра объекта по X
                t_center_x = (tx1 + tx2) / 2
                if lx1 <= t_center_x <= lx2:
                    is_inside = True

            if is_inside:
                candidate_lanes.append(lane)

        if candidate_lanes:
            # Если подходит несколько лэйнов, берем самый узкий (вложенный)
            if len(candidate_lanes) > 1:
                if direction == "lr":
                    candidate_lanes.sort(key=lambda l: (l["box"][3] - l["box"][1]))
                else:
                    candidate_lanes.sort(key=lambda l: (l["box"][2] - l["box"][0]))

            lane_text = candidate_lanes[0]["text"].strip()
            task["role"] = lane_text if lane_text else f"Lane-ID-{candidate_lanes[0]['class_id']}"

    return tasks


def show_annotated_objects(image, detections):
    try:
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        annotated = box_annotator.annotate(scene=image.copy(), detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections)
        sv.plot_image(annotated)
    except Exception as e:
        print(f"Visualization error: {e}")


# --- MAIN ---

print("Loading RFDETR model...")
model = RFDETRNano(pretrain_weights=WEIGHTS_PATH)
model.optimize_for_inference()

if hasattr(model, "id2label"):
    class_mapping = model.id2label
elif hasattr(model, "classes"):
    class_mapping = {i: name for i, name in enumerate(model.classes)}
else:
    print("Warning: Could not find class mapping in model. Using IDs.")
    class_mapping = {}

image = Image.open(IMAGE_PATH).convert("RGB")
width, height = image.size

direction = "lr" if width > height else "td"
print(f"Processing image {width}x{height}. Assumed direction: {direction.upper()}")

# 1. Детекция
detections = model.predict(image, threshold=THRESHOLD)
print(f"Detections found: {len(detections.xyxy)}")
# show_annotated_objects(image, detections) # Раскомментировать для показа картинки

# 2. OCR (для всех объектов)
all_results = qwen_ocr_from_boxes(image, detections, class_mapping, model_name=MODEL_NAME)

# --- DEBUG PRINT ---
print("\n--- DEBUG: ALL DETECTED OBJECTS ---")
for r in all_results:
    print(f"ID: {r['class_id']} | Text: '{r['text']}' | Box: {r['box']}")
print("-----------------------------------\n")

# 3. Сопоставление ролей
final_tasks = process_tasks_with_roles(all_results, direction)

# 4. Сортировка
ordered_tasks = order_actions(final_tasks, direction=direction)

print(f"\n--- FINAL RESULT ---")
for i, r in enumerate(ordered_tasks, 1):
    role_info = f" | Role: {r['role']}" if r.get('role') else ""
    # Выводим ID класса, чтобы можно было проверить правильность
    print(f"{i}. [Class ID: {r['class_id']}] {r['text']}{role_info}")

end = time.time()
print(f"\nExecution time: {end - start:.6f} seconds")
