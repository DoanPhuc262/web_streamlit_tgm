import streamlit as st
import numpy as np
import cv2
from PIL import Image
from pages.lib.common import set_background

# === Page Config ===
st.set_page_config(layout="centered")
st.title('🍎 Nhận dạng trái cây bằng YOLOv11')
set_background("Image/background1.jpg")

# === Constants ===
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1
BLUE = (255, 178, 50)
YELLOW = (0, 255, 255)

# === Classes ===
classes = ['Buoi', 'Cam', 'SauRieng', 'Tao', 'ThanhLong']

# === Load YOLO Model Once ===
if "Net" not in st.session_state:
    try:
        st.session_state["Net"] = cv2.dnn.readNet("D:/web/pages/yolo11n.onnx")
        
    except Exception as e:
        st.error(f"❌ Không thể load mô hình: {e}")

# === Drawing Utilities ===
def draw_label(im, label, x, y):
    dim, baseline = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)[0], 5
    cv2.rectangle(im, (x, y), (x + dim[0], y + dim[1] + baseline), (0, 0, 0), cv2.FILLED)
    cv2.putText(im, label, (x, y + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)

# === Preprocess ===
def pre_process(input_image, net):
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    return net.forward(net.getUnconnectedOutLayersNames())

# === Postprocess ===
def post_process(input_image, outputs):
    rows = outputs[0].shape[1]
    h, w = input_image.shape[:2]
    x_factor, y_factor = w / INPUT_WIDTH, h / INPUT_HEIGHT

    class_ids, confidences, boxes = [], [], []

    for r in range(rows):
        row = outputs[0][0][r]
        confidence = row[4]
        if confidence >= CONFIDENCE_THRESHOLD:
            class_scores = row[5:]
            class_id = np.argmax(class_scores)
            if class_scores[class_id] > SCORE_THRESHOLD:
                confidences.append(confidence)
                class_ids.append(class_id)
                cx, cy, bw, bh = row[0], row[1], row[2], row[3]
                left = int((cx - bw / 2) * x_factor)
                top = int((cy - bh / 2) * y_factor)
                width = int(bw * x_factor)
                height = int(bh * y_factor)
                boxes.append([left, top, width, height])

    # Prevent error when nothing is detected
    if len(boxes) == 0:
        cv2.putText(input_image, "❌ Không phát hiện vật thể nào", (20, 40),
                    FONT_FACE, FONT_SCALE, (0, 0, 255), THICKNESS, cv2.LINE_AA)
        return input_image

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    for i in indices.flatten():
        box = boxes[i]
        cv2.rectangle(input_image, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), BLUE, 3*THICKNESS)
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        draw_label(input_image, label, box[0], box[1])
    return input_image

# === Section: Image Upload and Prediction ===
st.header("📸 Nhận dạng từ ảnh tải lên")
img_file_buffer = st.file_uploader("Tải ảnh lên (bmp, png, jpg, jpeg)", type=["bmp", "png", "jpg", "jpeg"])

if img_file_buffer is not None:
    image = Image.open(img_file_buffer).convert("RGB")
    frame = np.array(image)
    frame = frame[:, :, ::-1]  # Convert RGB to BGR for OpenCV

    st.image(image, caption="Ảnh gốc", use_container_width=True)

    if st.button("🔍 Nhận dạng"):
        try:
            detections = pre_process(frame, st.session_state["Net"])
            result_img = post_process(frame.copy(), detections)
            t, _ = st.session_state["Net"].getPerfProfile()
            inf_time = f"Inference time: {t * 1000.0 / cv2.getTickFrequency():.2f} ms"
            cv2.putText(result_img, inf_time, (20, 40), FONT_FACE, FONT_SCALE, (0, 0, 255), THICKNESS, cv2.LINE_AA)
            st.image(result_img, caption="📌 Kết quả nhận dạng", channels="BGR", use_container_width=True)
        except Exception as e:
            st.error(f"❌ Lỗi xử lý ảnh: {e}")
