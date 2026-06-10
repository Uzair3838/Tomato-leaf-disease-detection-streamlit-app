# 🍅 Tomato Leaf Disease Detection System

A deep learning system that classifies **10 tomato leaf conditions** (9 diseases + healthy) from images with **~97.94% test accuracy** — built with a custom lightweight architecture and deployed as an interactive Streamlit web app.

> **Final Year Project** — BS Software Engineering, Islamia College University Peshawar

---

## 🎯 Problem Statement

Tomato crops are highly vulnerable to disease, causing major losses for farmers who often lack access to expert diagnosis. This system provides an automated solution — a farmer uploads a photo of a leaf and instantly receives a disease diagnosis along with recommended treatment.

---

## ✨ Key Features

- ✅ Classifies **10 conditions** — 9 diseases + healthy leaves
- ✅ **~97.94% test accuracy** on 6,400+ unseen images
- ✅ Custom architecture: **Inverted Residual Blocks + Efficient Channel Attention (ECA)**
- ✅ Only **~726 KB** model size — extremely lightweight
- ✅ **Streamlit web app** — upload an image, get instant results
- ✅ Displays **disease name, description, cure, and confidence score**
- ✅ Trained on **33,200 images** over 50 epochs with learning rate scheduling

---

## 🦠 Detected Conditions (10 Classes)

| # | Class |
|---|---|
| 1 | Bacterial Spot |
| 2 | Early Blight |
| 3 | Late Blight |
| 4 | Leaf Mold |
| 5 | Septoria Leaf Spot |
| 6 | Spider Mites (Two-spotted) |
| 7 | Target Spot |
| 8 | Tomato Yellow Leaf Curl Virus |
| 9 | Tomato Mosaic Virus |
| 10 | Healthy |

---

## 🏗️ Model Architecture

The model is a **custom lightweight CNN** inspired by MobileNetV2, built entirely from scratch — no pretrained weights used.

```
Input (224×224×3)
     │
     ▼
Conv2D Stem → BatchNorm → ReLU6
     │
     ▼
[Stage 1]  Inverted Residual Block (16→16, stride 1) + ECA
[Stage 2]  Inverted Residual Block (16→24, stride 2) + ECA
           Inverted Residual Block (24→24, stride 1) + ECA
[Stage 3]  Inverted Residual Block (24→40, stride 2) + ECA
[Stage 4]  Inverted Residual Block (40→96, stride 2) + ECA
     │
     ▼
Global Average Pooling
     │
     ▼
Dense(512, ReLU) → Dropout(0.3) → Dense(128, ReLU)
     │
     ▼
Dense(10, Softmax) → Predicted Class
```

**Efficient Channel Attention (ECA):** After each IRB stage, a 1D convolution captures cross-channel dependencies without fully connected layers — enabling stronger feature selection with virtually no extra parameters.

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Test Accuracy | **97.94%** |
| Best Validation Accuracy | **98.87%** (Epoch 47) |
| Model Parameters | **185,849** |
| Model Size | **~726 KB** |
| Training Images | 33,200 |
| Test Images | 6,431 |
| Input Size | 224 × 224 × 3 |
| Training Epochs | 50 |

---

## 🖥️ Streamlit App

The deployed app allows users to upload a tomato leaf image and instantly receive:

- 🦠 **Disease name**
- 📝 **Description** of the condition
- 💊 **Recommended cure / treatment**
- 📊 **Confidence score**

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Model Architecture | Custom IRB + ECA (TensorFlow / Keras) |
| Training Platform | Kaggle (Tesla T4 GPU) |
| Image Processing | NumPy, TensorFlow |
| Web App | Streamlit |
| Disease Database | JSON |

---

## 📁 Repository Structure

```
Tomato-leaf-disease-detection-streamlit-app/
│
├── app.py                        ← Streamlit web application
├── requirements.txt              ← Python dependencies
├── LICENSE
│
├── database/
│   └── disease_info.json         ← Disease descriptions & treatments
│
├── model/
│   └── Model9.h5                 ← Trained model weights (~726 KB)
│
└── training/                     ← (add your training notebook here)
    └── tomato_disease_training.ipynb
```

---

## 🚀 How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/Uzair3838/Tomato-leaf-disease-detection-streamlit-app.git
cd Tomato-leaf-disease-detection-streamlit-app
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`

---

## 📦 Dataset

- **Source:** [Kaggle — Tomato Leaf Disease Dataset](https://www.kaggle.com/datasets/uzyr008/tomato-leaf-disease)
- **Training images:** 33,200 (80/20 train/val split)
- **Test images:** 6,431
- **Classes:** 10 (9 diseases + healthy)

---

## 👤 Author

**Uzair Mustafa**
- GitHub: [@Uzair3838](https://github.com/Uzair3838)
- LinkedIn: [linkedin.com/in/uzair3838](https://www.linkedin.com/in/uzair3838)
- Email: uzair.mustafa3838@gmail.com

---

## 📄 License

This project is licensed under the [Apache 2.0 License](LICENSE).
