# 💳 End-to-End Fraud Detection System  
**Machine Learning · FastAPI · Streamlit · Production-Style Architecture**

---

## 📌 Overview

This repository contains a **production-inspired, end-to-end fraud detection system** built on a highly imbalanced real-world dataset.  
The project goes far beyond model training and demonstrates:

- Robust handling of **extreme class imbalance**
- Careful **metric selection** (PR-AUC, Recall, Precision)
- **Threshold tuning** driven by business constraints
- A **single source of inference** using FastAPI
- A **Streamlit frontend** acting as an internal fraud scoring tool
- Clean project structure and deployment-ready design

The goal is not just to predict fraud, but to show **how such systems are actually built and deployed in practice**.

---

## 🎯 Problem Statement

Credit card fraud detection is a classic **rare-event classification problem**:

- Fraudulent transactions are **< 0.2%** of all transactions
- Accuracy is a misleading metric
- False negatives are costly (missed fraud)
- False positives create customer friction

The challenge is to **maximize fraud capture** while keeping false alarms under control.

---

## 📊 Dataset

- **Source:** Kaggle – Credit Card Fraud Dataset (ULB)
- **Transactions:** 284,807  
- **Fraud rate:** ~0.17%

### Features
- `V1`–`V28`: **PCA-transformed, anonymized latent features**
- `Time`, `Amount`
- `Class`: target (1 = fraud, 0 = legitimate)

> ⚠️ **Important:**  
> Raw transaction details (merchant, country, device, etc.) are **not available**.  
> PCA is **pre-applied by the data provider**, reflecting how real payment systems expose only anonymized internal risk signals to models.

Because of this constraint, the UI is designed as an **internal fraud scoring tool**, not a customer-facing form.

### Dataset Access

Due to GitHub file size limits and best practices, the dataset is **not included**.

Download from:
[https://www.kaggle.com/mlg-ulb/creditcardfraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)

Place `creditcard.csv` inside:
```
fraud-detection/
```

---

## 🧠 Modeling Strategy

### 1️⃣ Logistic Regression (Baseline)
- Fails due to class imbalance
- Poor fraud recall

### 2️⃣ Class-Weighted Logistic Regression
- High recall
- Very low precision (too many false alarms)

### 3️⃣ Logistic Regression + SMOTE
- Improved minority-class representation
- Better PR-AUC
- Requires explicit threshold tuning

### 4️⃣ **Random Forest (Final Model)**
- Captures non-linear fraud patterns
- Excellent precision
- Best **PR-AUC ≈ 0.86**
- Chosen as the **final deployed model**

---

## 📐 Evaluation Metrics

Accuracy is intentionally **not** used for decision-making.

Key metrics:
- **Recall (Fraud Detection Rate)**
- **Precision (False Alarm Control)**
- **PR-AUC (Primary metric for imbalance)**
- **ROC-AUC (Ranking quality)**

---

## 🎚️ Threshold Tuning (Business-Aware)

Instead of relying on a fixed 0.5 cutoff:
- Fraud probability thresholds are **explicitly tunable**
- Allows dynamic trade-off between:
  - catching more fraud (higher recall)
  - reducing false positives (higher precision)

This mirrors real production fraud systems.

---

## 🏗️ System Architecture

```
Streamlit UI (Frontend)
│
│ HTTP / JSON
▼
FastAPI Inference Service  ←── Single Source of Truth
│
▼
Random Forest Model
```

### Why this design?
- No model duplication
- Consistent predictions across systems
- Easy upgrades & deployment
- Production-aligned separation of concerns

---

## 🔌 FastAPI Backend

### Responsibilities
- Load the trained model
- Perform **all inference**
- Apply configurable decision thresholds
- Serve batch and single predictions

### Endpoints

#### Health Check
```
GET /health
```

#### Prediction
```
POST /predict
```

**Request**
```json
{
  "features": [[...], [...]],
  "threshold": 0.6
}
```

**Response**
```json
{
  "fraud_probability": [0.87, 0.02],
  "prediction": [1, 0]
}
```

---

## 🖥️ Streamlit UI – Internal Fraud Scoring Engine

The Streamlit app acts as an **internal analyst / operations tool**.

### Supported Modes

1. **Manual Latent Feature Input**
   Used for analyst testing and debugging
2. **Random Transaction Demo**
   Samples a real transaction from the dataset
3. **CSV Upload (Batch Scoring)**
   Scores PCA-transformed transactions in bulk

### UI Features

* Adjustable probability threshold
* Batch fraud scoring
* Clear fraud / legitimate labeling
* Real-time inference via FastAPI

> Streamlit does **not** load the model — it always calls the backend API.

---

## 📂 Project Structure

```
ML-POS-SYSTEM/
│
├── fraud-detection/              # Main project directory
│   │
│   ├── api/                      # FastAPI backend service
│   │   ├── __pycache__/          # Python bytecode cache
│   │   └── app.py                # FastAPI application (inference endpoints)
│   │
│   ├── model/                    # Trained models and artifacts
│   │   ├── rf_fraud_model.pkl    # Trained Random Forest model (serialized)
│   │   └── feature_names.pkl     # Feature names for validation
│   │
│   ├── ui/                       # Streamlit frontend
│   │   └── streamlit_app.py      # Streamlit application (fraud scoring UI)
│   │
│   ├── initialize.ipynb          # Jupyter notebook (model training & experimentation)
│   │
│   └── creditcard.csv            # Dataset (not in repo; download separately)
│
├── .gitignore                    # Git ignore rules
├── .venv/                        # Python virtual environment (if present)
└── README.md                     # This file
```

### File Descriptions

| File/Directory | Purpose |
|----------------|---------|
| `fraud-detection/api/app.py` | FastAPI application serving inference endpoints (`/health`, `/predict`) |
| `fraud-detection/model/rf_fraud_model.pkl` | Serialized Random Forest model (trained and saved) |
| `fraud-detection/model/feature_names.pkl` | Feature names used for input validation |
| `fraud-detection/ui/streamlit_app.py` | Streamlit UI for interactive fraud scoring |
| `fraud-detection/initialize.ipynb` | Jupyter notebook containing model training, evaluation, and experimentation |
| `creditcard.csv` | Dataset file (must be downloaded separately from Kaggle) |

---

## 🚀 How to Run

### Prerequisites

- Python 3.8+
- Required packages: `fastapi`, `uvicorn`, `streamlit`, `scikit-learn`, `pandas`, `numpy`, `joblib`

### 1️⃣ Start the FastAPI Backend

```bash
uvicorn fraud-detection.api.app:app --reload
```

Verify:
- Health check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### 2️⃣ Start the Streamlit UI

```bash
cd fraud-detection/ui
streamlit run streamlit_app.py
```

Open:
```
http://localhost:8501
```

> **Note:** Make sure the FastAPI backend is running before starting the Streamlit app.

---

## 🧠 Key Technical Insights

* PCA features represent **internal risk signals**, not user inputs
* Fraud detection is a **decision system**, not just a classifier
* PR-AUC is the most meaningful metric for rare events
* Threshold tuning is as important as model choice
* Frontend and inference must be **strictly separated**

---

## 🔮 Possible Extensions

* SHAP / feature attribution for local explanations
* Authentication & rate limiting on API
* Dockerized deployment
* Cloud hosting (AWS / Render / Railway)
* Streaming inference (Kafka / real-time events)
* Model monitoring & drift detection

---

## 🏆 What This Project Demonstrates

✅ End-to-end ML pipeline  
✅ Imbalance-aware modeling  
✅ Business-driven evaluation  
✅ Production-style system design  
✅ API + UI integration  
✅ Clean, scalable architecture  

This project is designed to reflect **real-world ML engineering**, not just notebook experimentation.

---

## 👤 Author

**Shreyansh Singh**  
B.Tech, Chemical Engineering  
Indian Institute of Technology Delhi (IIT Delhi)

---
