import streamlit as st
from pages.lib.common import set_background, load_image
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# Page config and background
st.set_page_config(page_title="🔍 Dự đoán với YOLO", layout="wide")
set_background("Image/background1.jpg")

st.markdown("<h1 style='text-align: center;'>🎯 Dự đoán đối tượng bằng YOLOv8</h1>", unsafe_allow_html=True)
st.markdown("---")

# Load YOLO model
model = YOLO('yolo11n.pt', task='detect')

# Upload and color option
with st.sidebar:
    st.header("⚙️ Setting")
    uploaded_file = st.file_uploader("📤 Tải ảnh lên", type=['jpg', 'jpeg', 'png', 'bmp', 'webp', 'tif'])
    is_color = st.checkbox("🌈 Hiển thị ảnh màu", value=True)

imgin = None
imgout = None

# Display uploaded image
if uploaded_file:
    st.subheader("🖼️ Ảnh gốc")
    imgin = load_image(uploaded_file, is_color)
    st.image(imgin, caption="Ảnh gốc", channels="BGR" if is_color else "GRAY", use_container_width=True)
    st.markdown("---")

# Run prediction
if st.button("🚀 Dự đoán") and imgin is not None:
    with st.spinner("🔍 Đang dự đoán..."):
        names = model.names
        imgout = imgin.copy()
        results = model.predict(imgout, conf=0.6, verbose=False)
        annotator = Annotator(imgout)
        boxes = results[0].boxes.xyxy.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        confs = results[0].boxes.conf.tolist()

        for box, cls, conf in zip(boxes, clss, confs):
            annotator.box_label(
                box, 
                label=f"{names[int(cls)]} {conf:.2f}", 
                txt_color=(255, 0, 0), 
                color=(255, 255, 255)
            )
        imgout = annotator.result()

    st.success(f"✅ Dự đoán hoàn tất! Phát hiện {len(boxes)} đối tượng.")
    st.markdown("---")

    # Side-by-side image comparison
    st.markdown("### 📷 So sánh ảnh gốc và ảnh đã xử lý")
    col1, col2 = st.columns(2)
    with col1:
        st.image(imgin, caption="📌 Ảnh gốc", channels="BGR" if is_color else "GRAY", use_container_width=True)
    with col2:
        st.image(imgout, caption="📌 Kết quả dự đoán", channels="BGR", use_container_width=True)

    