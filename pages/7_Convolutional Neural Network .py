import os
import keras
import tensorflow as tf
from keras.models import load_model  # type: ignore
import streamlit as st
import numpy as np
from pages.lib.common import set_background

# UI setup
st.header('🧠 Convolutional Neural Network Model - Predict Objects')
set_background("Image/background1.jpg")
obj_names = ['Tomato', 'Walnut', 'daisy', 'rose', 'unknow']

# Load model
model = load_model('D:/web/pages/2kinds_ObjectNN.keras')

# Function to classify image
def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    label_index = np.argmax(result)
    confidence = np.round(max(result) * 100, 2)
    label = obj_names[label_index]
    return label, confidence, input_image

# Upload section
uploaded_file = st.file_uploader('📤 Vui lòng tải lên 1 hình ảnh', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Lưu ảnh tạm thời
    temp_path = os.path.join('Samples Test', uploaded_file.name)
    os.makedirs('Samples Test', exist_ok=True)
    with open(temp_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Dự đoán
    label, confidence, input_image = classify_images(temp_path)

    # Thông báo kết quả
    st.markdown(f"✅ Đây là: **{label}**  \n🎯 Độ chính xác: **{confidence}%**")

    # So sánh ảnh
    st.markdown("### 🪞 So sánh ảnh gốc và ảnh đã xử lý")
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption="📥 Ảnh gốc", use_container_width=True)
    with col2:
        st.image(input_image, caption=f"📤 Đã resize (180x180)", use_container_width=True)

    st.success(f"📋 Nhận diện hoàn tất — Phát hiện: **{label}**")
