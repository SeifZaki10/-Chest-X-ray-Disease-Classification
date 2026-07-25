import streamlit as st
import pandas as pd
from PIL import Image

from utils import load_model, predict
from labels import DISEASES


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Chest X-ray Disease Classifier",
    page_icon="🩻",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
model = load_model()

# -----------------------------
# Title
# -----------------------------
st.title("🩻 Chest X-ray Disease Classification")

st.markdown("""
Upload a **Chest X-ray image** and the model will predict the probability
of **14 thoracic diseases** using a **DenseNet121** deep learning model.
""")

st.divider()

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict"):

        with st.spinner("Analyzing image..."):

            predictions = predict(model, image)

        # Create DataFrame
        results = pd.DataFrame({
            "Disease": DISEASES,
            "Probability": predictions
        })

        results["Probability"] *= 100

        results = results.sort_values(
            by="Probability",
            ascending=False
        )

        st.success("Prediction Completed ✅")

        st.subheader("Top Predictions")

        st.dataframe(
            results.head(5).style.format({
                "Probability": "{:.2f}%"
            }),
            use_container_width=True
        )

        st.subheader("All Diseases")

        st.dataframe(
            results.style.format({
                "Probability": "{:.2f}%"
            }),
            use_container_width=True
        )

st.divider()

st.info(
    "⚠️ This application is for educational purposes only and "
    "must not be used for medical diagnosis."
)