
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🚧"
)

st.title("Road Damage Detection")
st.write("Upload a road image to detect cracks and road damage.")

model_path = Path(__file__).parent / "best(3).pt"

@st.cache_resource
def load_model():
    return YOLO(str(model_path))

try:
    model = load_model()
except Exception as e:
    st.error(f"Model load nahi hua: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Road ki image upload karein",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    if st.button("Detect Road Damage"):
        with st.spinner("Road damage detect ho raha hai..."):
            results = model.predict(
                source=np.array(image),
                conf=0.10
            )

            output_image = results[0].plot()[:, :, ::-1]

        st.subheader("Detection Result")
        st.image(output_image, use_container_width=True)

        if len(results[0].boxes) > 0:
            st.success(
                f"{len(results[0].boxes)} damage detections mili hain."
            )

            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])

                st.write(
                    f"{class_name} — Confidence: {confidence:.2%}"
                )
        else:
            st.warning(
                "Koi damage detect nahi hua. Doosri road image try karein."
            )

        st.subheader("Detected Road Damage")
        st.image(output_image)

        st.write("Detection complete!")
