# common.py
import base64
import streamlit as st
import cv2
import numpy as np

def set_background(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def load_image(uploaded_file, is_color=True):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    mode = cv2.IMREAD_COLOR if is_color else cv2.IMREAD_GRAYSCALE
    return cv2.imdecode(file_bytes, mode)

def show_image_output(title, img):
    st.image(img, caption=title, use_container_width=True)
