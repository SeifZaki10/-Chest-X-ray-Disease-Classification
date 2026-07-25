import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input
import streamlit as st


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_model.keras")
    return model


def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    image = np.array(image, dtype=np.float32)

    image = preprocess_input(image)

    image = np.expand_dims(image, axis=0)

    return image


def predict(model, image):
    processed = preprocess_image(image)
    preds = model.predict(processed, verbose=0)
    return preds[0]