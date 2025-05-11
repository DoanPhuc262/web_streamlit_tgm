import streamlit as st
from pages.lib.common import set_background, load_image
import cv2
import numpy as np
import time

# Page configuration and background
st.set_page_config(page_title="Nhận diện hình dạng", layout="wide")
set_background("Image/background1.jpg")

# Title and separator
st.markdown(
    "<h1 style='text-align: center; color: white;'>🔷 Nhận diện hình dạng</h1>",
    unsafe_allow_html=True
)
st.markdown("<hr style='border: 1px solid white;'>", unsafe_allow_html=True)

# Sidebar for image input
with st.sidebar:
    st.header("⚙️ Setting")
    uploaded_file = st.file_uploader("📤 Tải ảnh lên", type=['jpg', 'jpeg', 'png', 'bmp', 'webp', 'tif'])
    process_button = st.button("🚀 Nhận diện hình dạng")

# Thresholding function
def phan_nguong(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    for x in range(M):
        for y in range(N):
            r = imgin[x, y]
            s = 255 if r == 63 else 0
            imgout[x, y] = np.uint8(s)
    return cv2.medianBlur(imgout, 7)

# Process image if uploaded
if uploaded_file:
    imgin = load_image(uploaded_file, is_color=True)

    if process_button:
        with st.spinner("⏱️ Đang xử lý..."):
            time.sleep(1)  # Simulate processing delay

            imggray = cv2.cvtColor(imgin, cv2.COLOR_RGB2GRAY)
            binary = phan_nguong(imggray)
            m = cv2.moments(binary)
            hu = cv2.HuMoments(m)
            h0 = hu[0, 0]
            imgout = imgin.copy()

            # Determine shape
            if 0.000622 <= h0 <= 0.000628:
                shape = ' Circle'
            elif 0.000646 <= h0 <= 0.000666:
                shape = ' Square'
            elif 0.000727 <= h0 <= 0.000749:
                shape = 'Triangle'
            else:
                shape = ' Unknown'

            cv2.putText(imgout, shape, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Output comparison in 2 columns
        st.markdown("### 🪞 So sánh kết quả")
        col1, col2 = st.columns(2)
        with col1:
            st.image(imgin, caption="📥 Ảnh gốc", channels="BGR", use_container_width=True)
        with col2:
            st.image(imgout, caption=f"📤 Kết quả: {shape}", channels="BGR", use_container_width=True)

        # Summary
        st.success(f"📋 Nhận diện hoàn tất — Phát hiện: **{shape}**")
    else:
        st.info("💡 Nhấn nút **'Nhận diện hình dạng'** sau khi tải ảnh.")
