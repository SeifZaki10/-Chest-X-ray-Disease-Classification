import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image

from utils import load_model, predict
from labels import DISEASES


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Chest X-ray Disease Classifier",
    page_icon="🩻",
    layout="centered"
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------
model = load_model()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:

    st.header("🧠 Model Information")

    st.write("**Architecture:** DenseNet121")
    st.write("**Task:** Multi-label Classification")
    st.write("**Classes:** 14 Thoracic Diseases")
    st.write("**Transfer Learning:**")
    st.write("- Feature Extraction")
    st.write("- Fine-Tuning")
    st.write("**Framework:** TensorFlow / Keras")
    st.write("**Deployment:** Streamlit")

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("🩻 Chest X-ray Disease Classification")

st.markdown("""
Upload a **Chest X-ray image** to predict the probability of **14 thoracic diseases**
using a **DenseNet121 Transfer Learning** model.
""")

st.divider()

# --------------------------------------------------
# Upload Image
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Chest X-ray",
        use_container_width=True
    )

    if st.button("🔍 Analyze Chest X-ray", use_container_width=True):

        with st.spinner("Analyzing image..."):

            predictions = predict(model, image)

        # ------------------------------------------
        # Results DataFrame
        # ------------------------------------------
        results = pd.DataFrame({
            "Disease": DISEASES,
            "Probability": predictions * 100
        })

        results = results.sort_values(
            by="Probability",
            ascending=False
        ).reset_index(drop=True)

        st.success("Prediction Completed Successfully ✅")

        # ------------------------------------------
        # Top Predictions
        # ------------------------------------------
        st.subheader("🏆 Top Predictions")

        cols = st.columns(5)

        for i in range(5):

            with cols[i]:

                st.metric(
                    label=results.loc[i, "Disease"],
                    value=f"{results.loc[i, 'Probability']:.2f}%"
                )

        # ------------------------------------------
        # Probability Chart
        # ------------------------------------------
        st.subheader("📊 Prediction Probabilities")

        fig = px.bar(
            results,
            x="Probability",
            y="Disease",
            orientation="h",
            color="Probability",
            color_continuous_scale="Blues",
            text="Probability"
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        fig.update_layout(
            height=650,
            coloraxis_showscale=False,
            xaxis_title="Probability (%)",
            yaxis_title="Disease",
            yaxis=dict(categoryorder="total ascending")
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ------------------------------------------
        # All Diseases
        # ------------------------------------------
        st.subheader("📋 All Diseases")

        st.dataframe(
            results.style.format({
                "Probability": "{:.2f}%"
            }),
            use_container_width=True
        )

st.divider()

st.caption("Developed by Seif Zaki")

st.info(
    "⚠️ This application is intended for educational purposes only and should not be used for medical diagnosis."
)