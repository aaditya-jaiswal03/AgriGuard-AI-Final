# 🌿 AgriGuard AI — Smart Crop Health Assistant

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://agrigaurd-ai-crop-disease-detection.streamlit.app/)

AgriGuard AI is a lightweight computer vision tool built to help farmers and agronomists catch plant diseases early. By uploading a single photo of a suspect leaf, the application runs an inference pass against a custom Convolutional Neural Network (CNN) to instantly diagnose the pathology and provide actionable next steps for crop recovery.

---

## 🎯 What It Does
* **Instant Pathology Detection:** Identifies healthy vs. diseased conditions across **38 distinct crop categories**.
* **High-Reliability Inference:** Achieves **~95% validation accuracy** on standard out-of-sample visual evaluations.
* **Field-Ready UI:** Built with a responsive, de-jargonized interface optimized for mobile usage outdoors.
* **Portable Reports:** Allows users to download clean text summaries of diagnoses to show local agricultural suppliers.

---

## 🖥️ System Architecture & Tech Stack
The project decouples raw model compilation from frontend delivery to ensure sub-second cold boot times on headless server containers.

* **Frontend Engine:** Python native UI via [Streamlit](https://streamlit.io/)
* **Deep Learning Framework:** TensorFlow 2.x / Keras
* **Array Operations & Buffer Parsing:** NumPy, Pillow (PIL)
* **Storage Layer:** Git Large File Storage (LFS) for remote `.keras` matrix hosting

---

## 📊 Dataset & Training Telemetry
The underlying model weights were trained using highly augmented visual samples to account for real-world orientation and scaling variance.

* **Base Corpus:** [Kaggle PlantVillage Pathology Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
* **Volume:** 87,000+ labeled RGB source sequences
* **Input Topology:** Normalized to `128x128` multidimensional tensors
* **Augmentation Pipeline:** Automated horizontal/vertical reflections, random scaling, and rotational skews

---

## 🚀 Local Development Setup

### 1. Clone the Repository
Ensure you have Git LFS initialized locally so the heavy `.keras` weights pull cleanly.
```bash
git clone [https://github.com/aaditya-jaiswal03/AgriGuard-AI-Final.git](https://github.com/aaditya-jaiswal03/AgriGuard-AI-Final.git)
cd AgriGuard-AI-Final

