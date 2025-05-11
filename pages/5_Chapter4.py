import streamlit as st
from pages.lib.common import set_background, load_image, show_image_output
import pages.lib.chapter4 as c4
import cv2
import numpy as np
import time

# Page config and background
st.set_page_config(page_title="Xử lý ảnh Chapter 4", layout="wide")
set_background("Image/background1.jpg")

# Title
st.markdown("<h1 style='text-align: center; color: white;'>🎨 Xử lý ảnh - Chapter 4</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid white;'>", unsafe_allow_html=True)

# Sidebar: Upload and Settings
with st.sidebar:
    st.header("⚙️ Setting")
    uploaded_file = st.file_uploader("📤 Tải ảnh lên", type=['jpg', 'jpeg', 'png', 'bmp', 'webp', 'tif'])
    is_color = st.checkbox("🌈 Mở ảnh màu", value=True)
    option = st.selectbox("🎛️ Chọn chức năng xử lý ảnh", [
        "Spectrum", "Remove Moire Simple", "Remove Moire", "Remove Interference"
    ])
    process = st.button("🚀 Xử lý")

# Process
imgin = None
imgout = None

if uploaded_file:
    imgin = load_image(uploaded_file, is_color)

    if process:
        with st.spinner("⏱️ Đang xử lý ảnh..."):
            time.sleep(1)  # Simulate processing time
            if option == "Spectrum":
                imgout = c4.Spectrum(imgin)
            elif option == "Remove Moire Simple":
                imgout = c4.RemoveMoireSimple(imgin)
            elif option == "Remove Moire":
                imgout = c4.RemoveMoire(imgin)
            elif option == "Remove Interference":
                imgout = c4.RemoveInterference(imgin)
        
        # Show original and result
        st.markdown("### 🪞 So sánh ảnh gốc và ảnh đã xử lý")
        col1, col2 = st.columns(2)
        with col1:
            st.image(imgin, caption="📥 Ảnh gốc", channels="BGR" if is_color else "GRAY",use_container_width=True)
        with col2:
            st.image(imgout, caption=f"📤 Kết quả: {option}", use_container_width=True)

        st.success(f"📋 Hoàn tất xử lý bằng phương pháp: **{option}**")
    else:
        st.image(imgin, caption="📥 Ảnh gốc", channels="BGR" if is_color else "GRAY", use_container_width=True)
        st.info("💡 Vui lòng nhấn **Xử lý** để bắt đầu.")
