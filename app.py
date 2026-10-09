
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

st.title("Road Damage Detection")
st.write("Upload a road image to detect road damage.")

model_path = os.path.join(os.path.dirname(__file__), "best.pt")
model = YOLO(model_path)

uploaded_file = st.file_uploader(
    "Upload Road Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Image")
    st.image(image)

    if st.button("Detect Road Damage"):
        results = model.predict(image, conf=0.10)

        output_image = results[0].plot()[:, :, ::-1]

        st.subheader("Detected Road Damage")
        st.image(output_image)

        st.write("Detection complete!")
