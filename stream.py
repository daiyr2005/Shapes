import streamlit as st
import requests
from PIL import Image

st.title("Image")

file = st.file_uploader("image", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file)
    st.image(img, caption="Image")

    if st.button("Predict"):
        response = requests.post(
            "http://127.0.0.1:8002/predict",
            files={"image": (file.name, file.getvalue(), file.type)}
        )

        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: {result['Answer']}")
        else:
            st.error(response.text)