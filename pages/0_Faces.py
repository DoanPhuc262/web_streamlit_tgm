import streamlit as st
import numpy as np
import cv2 as cv
import joblib
from PIL import Image
from pages.lib.common import set_background

#Setup Streamlit interface
st.set_page_config(layout="centered")
st.title("Recognize faces by webcam & and image")
set_background("Image/background1.jpg")

# ===== Load model  =====
svc = joblib.load('D:/web/pages/svc.pkl')

labels = ['Loc', 'Binh', 'Dat', 'Doan', 'Lap', 'Loi']

detector = cv.FaceDetectorYN.create(
    'D:/web/pages/face_detection_yunet_2023mar.onnx', '', (320, 320), score_threshold=0.9)
recognizer = cv.FaceRecognizerSF.create('D:/web/pages/face_recognition_sface_2021dec.onnx', '')

# function detect 
def recognize_and_draw(frame, faces, threshold=0.65):
    if faces[1] is not None:
        for face in faces[1]:
            coords = face[:-1].astype(np.int32)
            face_align = recognizer.alignCrop(frame, face)
            face_feature = recognizer.feature(face_align)

            probs = svc.predict_proba(face_feature)[0]
            confidence = np.max(probs)
            pred = np.argmax(probs)

            if confidence >= threshold:
                name = labels[pred]
            else:
                name = "Unknown"

            cv.rectangle(frame, (coords[0], coords[1]), 
                         (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), 2)
            cv.putText(frame, f"{name} ({confidence:.2f})", 
                       (coords[0], coords[1]-10), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    else:
        cv.putText(frame, "No faces found", (10, 30), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame


#Webcam real-time 
st.subheader("Recognize faces by webcam")

if 'run' not in st.session_state:
    st.session_state.run = False

start = st.button("🔴 Turn On Webcam")
stop = st.button("⏹️ Turn Off Webcam")

FRAME_WINDOW = st.image([])

if start:
    st.session_state.run = True
if stop:
    st.session_state.run = False

cap = cv.VideoCapture(0)

if st.session_state.run:
    detector.setInputSize((int(cap.get(3)), int(cap.get(4))))
    frame_placeholder = st.empty()

    while st.session_state.run:
        success, frame = cap.read()
        if not success:
            st.warning("Inaccessible webcam")
            break
        faces = detector.detect(frame)
        frame = recognize_and_draw(frame, faces)
        frame_placeholder.image(frame, channels="BGR")

    cap.release()

# Upload 
st.title("📸 Recognize faces by upload image")
uploaded_file = st.file_uploader("Chọn ảnh (jpg/jpeg/png)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv.imdecode(file_bytes, cv.IMREAD_COLOR)

    if image is not None:
        st.image(image, channels='BGR', caption="Ảnh gốc", use_container_width=True)

        # Resize image to fit with input size của detector
        detector.setInputSize((image.shape[1], image.shape[0]))
        faces = detector.detect(image)

        if faces[1] is not None:
            for face in faces[1]:
                coords = face[:-1].astype(np.int32)
                face_align = recognizer.alignCrop(image, face)
                face_feature = recognizer.feature(face_align)

                probs = svc.predict_proba(face_feature)[0]
                confidence = np.max(probs)
                pred = np.argmax(probs)

                if confidence >= 0.65:
                    name = labels[pred]
                else:
                    name = "Unknown"

                cv.rectangle(image, (coords[0], coords[1]),
                             (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), 2)
                cv.putText(image, f"{name} ({confidence:.2f})",
                           (coords[0], coords[1] - 10), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        else:
            cv.putText(image, "No faces found", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        st.markdown("### 📷 So sánh ảnh gốc và ảnh đã dự đoán")
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv.imdecode(file_bytes, cv.IMREAD_COLOR), caption="📌 Ảnh gốc", channels="BGR", use_container_width=True)
        with col2:
            st.image(image, caption="📌 Kết quả dự đoán", channels="BGR", use_container_width=True)
    else:
        st.error("Không thể đọc ảnh.")
