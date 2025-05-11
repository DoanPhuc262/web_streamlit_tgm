# pages/5_XuLyAnhChapter9.py
import streamlit as st
from pages.lib.common import set_background, load_image
import cv2
import numpy as np
import pages.lib.chapter9 as c9
import time

# Page config and background
st.set_page_config(page_title="Xử lý ảnh Chapter 9", layout="wide")
set_background("Image/background1.jpg")

# Title
st.markdown("<h1 style='text-align: center; color: white;'>🧩 Xử lý ảnh - Chapter 9</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid white;'>", unsafe_allow_html=True)

# Sidebar: Upload & Controls
with st.sidebar:
    st.header("⚙️ Setting")
    uploaded_file = st.file_uploader("📤 Tải ảnh lên", type=['jpg', 'jpeg', 'png', 'bmp', 'webp', 'tif'])
    option = st.selectbox("🎛️ Chọn chức năng xử lý Chapter 9", [
        "Connected Components", "Remove Small Rice"
    ])
    process = st.button("🚀 Xử lý ảnh")

# Main content
imgin = None
imgout = None

if uploaded_file:
    imgin = load_image(uploaded_file, is_color=True)

    if process:
        with st.spinner("⏱️ Đang xử lý ảnh..."):
            time.sleep(1)  # Simulate delay
            imggray = cv2.cvtColor(imgin, cv2.COLOR_RGB2GRAY)

            if option == "Connected Components":
                imgout = c9.ConnectedComponents(imggray)
            elif option == "Remove Small Rice":
                imgout = c9.RemoveSmallRice(imggray)

        # Comparison layout
        st.markdown("### 🪞 So sánh ảnh gốc và ảnh đã xử lý")
        col1, col2 = st.columns(2)
        with col1:
            st.image(imgin, caption="📥 Ảnh gốc", channels="BGR", use_column_width=True)
        with col2:
            st.image(imgout, caption=f"📤 Kết quả: {option}", use_column_width=True)

        st.success(f"📋 Hoàn tất xử lý với phương pháp: **{option}**")
    else:
        st.image(imgin, caption="📥 Ảnh gốc", channels="BGR", use_column_width=True)
        st.info("💡 Vui lòng chọn chức năng và nhấn **Xử lý ảnh** để tiếp tục.")
