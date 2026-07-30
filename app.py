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
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Model Information")

    st.write("**Architecture:** DenseNet121")
    st.write("**Task:** Multi-label Classification")
    st.write("**Classes:** 14 Thoracic Diseases")
    st.write("**Framework:** TensorFlow / Keras")
    st.write("**Transfer Learning:** Feature Extraction + Fine-Tuning")
    st.write("**Deployment:** Streamlit")

# -----------------------------
# Title
# -----------------------------
st.title("🩻 Chest X-ray Disease Classification")

st.markdown("""
Upload a **Chest X-ray image** to receive the predicted probabilities for **14 thoracic diseases**
using a **DenseNet121 Transfer Learning** model.
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
        caption="Uploaded Chest X-ray",
        width=400
    )

    if st.button("🔍 Analyze Chest X-ray"):

        with st.spinner("Analyzing image..."):

            predictions = predict(model, image)

        results = pd.DataFrame({
            "Disease": DISEASES,
            "Probability": predictions * 100
        })

        results = results.sort_values(
            by="Probability",
            ascending=False
        )

        st.success("Prediction Completed Successfully ✅")

        st.subheader("Top Predictions")

        st.caption(
            "The table below shows the five diseases with the highest predicted probabilities."
        )

        st.dataframe(
            results.head(5).style.format({
                "Probability": "{:.2f}%"
            }),
            use_container_width=True
        )

        if results.iloc[0]["Probability"] < 20:
            st.warning(
                "No disease was predicted with high confidence."
            )

st.divider()

st.caption("Developed by Seif Zaki")

st.info(
    "⚠️ This application is intended for educational purposes only and should not be used for medical diagnosis."
)