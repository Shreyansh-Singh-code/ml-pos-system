# 💳 Fraud Detection System (End-to-End ML + API + UI)

An end-to-end **fraud detection system** built using **machine learning**, **FastAPI**, and **Streamlit**, designed to mirror **real-world payment fraud architectures**.

This project demonstrates:
- Handling **extreme class imbalance**
- Training and evaluating multiple ML models
- **Threshold tuning based on business constraints**
- **Production-style system design** with a single inference backend
- An **internal fraud scoring UI** for analysts and batch evaluation

---

## 📌 Problem Statement

Fraud detection is a **highly imbalanced classification problem**, where fraudulent transactions form a tiny fraction of total transactions. Traditional accuracy-based evaluation fails, and systems must be designed to:

- Maximize fraud capture (high recall)
- Minimize false alarms (precision)
- Support **real-time and batch scoring**
- Maintain **consistent inference across systems**

---

## 📊 Dataset

- **Source**: Credit Card Fraud Dataset (Kaggle)
- **Size**: 284,807 transactions
- **Fraud Rate**: ~0.17%
- **Features**:
  - `V1`–`V28`: PCA-transformed, anonymized latent features
  - `Amount`, `Time`
  - `Class`: Target (1 = fraud, 0 = legitimate)

## Dataset Access

Due to GitHub file size limits, the dataset is not included in this repository.

Please download the dataset from:
https://www.kaggle.com/mlg-ulb/creditcardfraud

Place `creditcard.csv` inside:
`fraud-detection/`


⚠️ **Important Constraint**  
Raw transaction attributes (merchant, country, device, etc.) are **not available**. PCA features are **pre-applied by the data provider**, reflecting how real payment systems expose only anonymized internal signals to models.

---

## 🧠 Modeling Approach

### 1️⃣ Baseline: Logistic Regression
- Poor fraud recall due to imbalance

### 2️⃣ Class-Weighted Logistic Regression
- Improved recall
- Very high false positive rate

### 3️⃣ Logistic Regression + SMOTE
- Better fraud representation in training space
- Improved PR-AUC
- Required **decision threshold tuning**

### 4️⃣ Random Forest (Final Model)
- Captures **non-linear fraud patterns**
- Strong precision without aggressive resampling
- Best **PR-AUC (≈ 0.86)**

📌 **Final deployed model**: **Random Forest**

---

## 📐 Evaluation Metrics (Why Not Accuracy?)

Due to severe imbalance:
- **Accuracy is misleading**
- Model selection is based on:
  - **Recall (Fraud Detection Rate)**
  - **Precision (False Alarm Control)**
  - **PR-AUC** (most important)
  - **ROC-AUC**

---

## 🎚️ Threshold Tuning (Business-Aware)

Instead of using the default 0.5 cutoff:
- Decision thresholds are **explicitly tunable**
- Allows trade-off between:
  - catching more fraud (recall)
  - reducing customer friction (precision)

This reflects **real production systems**, where thresholds are adjusted per business policy.

---

## 🏗️ System Architecture

