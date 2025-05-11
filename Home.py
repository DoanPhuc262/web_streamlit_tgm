import streamlit as st
from PIL import Image
from pages.lib.common import set_background

# Page config and background
st.set_page_config(page_title="Digital Image Processing", layout="wide")
set_background("Image/background.jpg")  # Make sure this image path exists

# Title
st.markdown(
    "<h1 style='text-align: center; color: white; padding-top: 20px;'>📷 Digital Image Processing</h1>",
    unsafe_allow_html=True
)
st.markdown("<hr style='border: 2px solid white;'>", unsafe_allow_html=True)

def load_and_resize(image_path, size=(200, 200)):
    img = Image.open(image_path)
    img = img.resize(size)
    return img
#content
col1, col2, col3 = st.columns(3)
with col1:
    st.image(load_and_resize("Image/h0.jpg"), use_container_width=True)
    st.markdown("<center><b>Recognize Faces</b></center>", unsafe_allow_html=True)
with col2:
    st.image(load_and_resize("Image/h1.jpg"), use_container_width=True)
    st.markdown("<center><b>Yolov11 Detect Objects</b></center>", unsafe_allow_html=True)
with col3:
    st.image(load_and_resize("Image/h2.jpg"), use_container_width=True)
    st.markdown("<center><b>Recognize Shapes</b></center>", unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)
with col4:
    st.image(load_and_resize("Image/h3.jpg"), use_container_width=True)
    st.markdown("<center><b>Image Processing</b></center>", unsafe_allow_html=True)
with col5:
    st.image(load_and_resize("Image/h4.jpg"), use_container_width=True)
    st.markdown("<center><b>Detect Objects by CNN</b></center>", unsafe_allow_html=True)
with col6:
    st.image(load_and_resize("Image/h5.jpg"), use_container_width=True)
    st.markdown("<center><b>Others</b></center>", unsafe_allow_html=True)

# Footer
st.markdown("<hr style='border: 1px solid white;'>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: lightgray;'>Built with ❤️ using Streamlit</p>",
    unsafe_allow_html=True
)
