import streamlit as st
from pages.lib.common import set_background, load_image, show_image_output
import cv2
import numpy as np
import pages.lib.chapter3 as c3
import time

# Page setup and background
st.set_page_config(page_title="Xử lý ảnh Chapter 3", layout="wide")
set_background("Image/background1.jpg")

# Title
st.markdown("<h1 style='text-align: center; color: white;'>🛠️ Xử lý ảnh - Chapter 3</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid white;'>", unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Setting")
    uploaded_file = st.file_uploader("📤 Tải ảnh lên", type=['jpg', 'jpeg', 'png', 'bmp', 'webp', 'tif'])
    option = st.selectbox("🎛️ Chọn chức năng xử lý ảnh", [
        "Negative", "NegativeColor", "Logarit", "Power", 
        "PiecewiseLine", "Histogram", "HistEqual", "HistEqualColor",
        "LocalHist", "HistStat", "SmoothBox", "SmoothGauss", 
        "MedianFilter", "Sharp", "Gradient"
    ])
    process_button = st.button("🚀 Xử lý ảnh")

# Main processing logic
if uploaded_file:
    imgin = load_image(uploaded_file, is_color=True)

    if process_button:
        with st.spinner("⏱️ Đang xử lý ảnh..."):
            time.sleep(1)  # Simulate processing time
            imggray = cv2.cvtColor(imgin, cv2.COLOR_RGB2GRAY)

            # Map option to function
            if option == "Negative":
                imgout = c3.Negative(imggray)
            elif option == "NegativeColor":
                imgout = c3.NegativeColor(imgin)
            elif option == "Logarit":
                imgout = c3.logarit(imggray)
            elif option == "Power":
                imgout = c3.Power(imggray)
            elif option == "PiecewiseLine":
                imgout = c3.PiecewiseLine(imggray)
            elif option == "Histogram":
                imgout = c3.Histogram(imggray)
            elif option == "HistEqual":
                imgout = cv2.equalizeHist(imggray)
            elif option == "HistEqualColor":
                imgout = c3.HistEqualColor(imgin)
            elif option == "LocalHist":
                imgout = c3.LocalHist(imggray)
            elif option == "HistStat":
                imgout = c3.HistStat(imggray)
            elif option == "SmoothBox":
                imgout = cv2.boxFilter(imggray, cv2.CV_8UC1, (21, 21))
            elif option == "SmoothGauss":
                imgout = cv2.GaussianBlur(imggray, (43, 43), 7.0)
            elif option == "MedianFilter":
                imgout = cv2.medianBlur(imggray, 3)
            elif option == "Sharp":
                imgout = c3.Sharp(imggray)
            elif option == "Gradient":
                imgout = c3.Gradient(imggray)

        # Side-by-side image comparison
        st.markdown("### 🪞 So sánh ảnh gốc và ảnh đã xử lý")
        col1, col2 = st.columns(2)
        with col1:
            st.image(imgin, caption="📥 Ảnh gốc", channels="BGR", use_container_width=True)
        with col2:
            st.image(imgout, caption=f"📤 Kết quả: {option}", use_container_width=True)

        st.success(f"📋 Hoàn tất xử lý ảnh bằng phương pháp: **{option}**")
    else:
        st.info("💡 Vui lòng chọn phương pháp và nhấn **Xử lý ảnh**.")
