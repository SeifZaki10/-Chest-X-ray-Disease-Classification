# 🩻 Chest X-ray Disease Classification

A deep learning web application for multi-label chest X-ray disease classification using **DenseNet121** and **Streamlit**.

---

## Features

- Predicts **14 thoracic diseases**
- Uses a pretrained **DenseNet121**
- Image preprocessing with TensorFlow
- Interactive Streamlit interface
- Displays Top-5 predicted diseases
- Shows prediction probabilities

---

## Diseases

- No Finding
- Enlarged Cardiomediastinum
- Cardiomegaly
- Lung Opacity
- Lung Lesion
- Edema
- Consolidation
- Pneumonia
- Atelectasis
- Pneumothorax
- Pleural Effusion
- Pleural Other
- Fracture
- Support Devices

---

## Model

- DenseNet121
- Image Size: 224×224
- Transfer Learning
- Binary Crossentropy Loss
- Adam Optimizer

---

## Run Locally

```bash
pip install -r requirements.txt

streamlit run app.py
```

---

## Disclaimer

This project is intended for educational purposes only and should **not** be used for real medical diagnosis.