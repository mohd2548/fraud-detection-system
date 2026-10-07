# Fraud Detection & Transaction Risk Scoring System

An end-to-end fraud investigation system built using **Python, Scikit-learn, SHAP, RAG-based investigation guidance, and Flask**.

The system combines machine-learning fraud prediction with explainability, evidence retrieval, and human investigator review.

---

## 1. Project Overview

Financial transaction fraud detection is a highly imbalanced and time-dependent classification problem.

This project builds a complete fraud investigation workflow:

```text
Transaction
    ↓
Feature Engineering
    ↓
Random Forest
    ↓
Fraud Prediction + Risk Score
    ↓
SHAP Explainability
    ↓
Transaction Evidence
    ↓
RAG Investigation Evidence
    ↓
Investigation Recommendation
    ↓
Human Investigator Decision
```

The goal is not only to predict whether a transaction is potentially fraudulent, but also to provide evidence and explanations that can support an investigator's decision.

---

## 2. Key Features

- Chronological train/validation/test split
- Leakage-aware historical behavioural features
- Fraud-focused feature engineering
- Class imbalance handling
- Random Forest fraud classifier
- Probability-based risk scoring
- SHAP-based local explanations
- Evidence-based local RAG investigation guidance
- Human-in-the-loop investigation decision
- Flask web application
- Saved ML model and preprocessing artifacts
- Dynamic transaction evidence on the result page

---

## 3. Dataset

The project uses the **PAYSim synthetic mobile-money transaction dataset**.

Original dataset:

- Rows: **6,362,620**
- Columns: **11**
- Fraudulent transactions: **8,213**
- Non-fraudulent transactions: **6,354,407**
- Fraud prevalence: approximately **0.129%**

Original columns include:

```text
step
type
amount
nameOrig
oldbalanceOrg
newbalanceOrig
nameDest
oldbalanceDest
newbalanceDest
isFraud
isFlaggedFraud
```

PAYSim is a synthetic dataset. Therefore, model performance on this dataset should not be interpreted as evidence of equivalent performance on real-world banking data.

---

## 4. Time-Based Data Split

Instead of randomly splitting transactions, the project uses chronological periods:

| Dataset | Steps | Shape |
|---|---:|---:|
| Train | 1–520 | 6,082,007 |
| Validation | 521–631 | 191,147 |
| Test | 632–743 | 89,466 |

The chronological design better represents a production scenario where historical transactions are used to predict future transactions.

---

## 5. Historical Behaviour Features

Historical transaction behaviour was calculated without using future information.

### Sender behaviour

- `sender_past_transaction_count`
- `sender_past_total_amount`
- `sender_past_avg_amount`
- `sender_past_unique_receivers`

### Receiver behaviour

- `receiver_past_transaction_count`
- `receiver_past_total_amount`
- `receiver_past_avg_amount`
- `receiver_past_unique_senders`

These features help the model compare the current transaction with previously observed behaviour.

---

## 6. Feature Engineering

Transaction-level features include:

- Origin balance error
- Absolute origin balance error
- Amount-to-origin balance ratio
- Sender balance depletion
- Destination balance depletion
- Origin balance zero indicator
- Sender balance change
- Receiver balance change
- Receiver amount difference
- Transaction hour
- Day of week
- Weekend indicator
- Transaction type one-hot encoding

The final ML input contains **32 features**.

---

## 7. Class Imbalance Handling

Fraudulent transactions represent only about **0.129%** of the original PAYSim dataset.

Training was therefore balanced using:

- All available fraudulent training transactions: **5,781**
- Non-fraudulent training sample: **28,905**
- Total balanced training sample: **34,686**
- Fraud proportion in balanced sample: approximately **16.67%**

The validation and test datasets were kept in their chronological distributions.

---

## 8. Machine Learning Model

The selected model is:

**RandomForestClassifier**

The model was selected after model comparison and evaluated using fraud-focused metrics.

The model produces:

1. A binary fraud prediction
2. A probability-based fraud risk score

The application uses the probability of the fraud class as the displayed risk score.

---

## 9. Model Performance

### Validation

| Metric | Result |
|---|---:|
| Precision | 1.0000 |
| Recall | 0.9992 |
| F1 Score | 0.9996 |
| False Positives | 0 |
| False Negatives | 1 |

### Final Chronological Test

| Metric | Result |
|---|---:|
| Precision | 1.0000 |
| Recall | 0.9992 |
| F1 Score | 0.9996 |
| PR-AUC | 0.999996 |
| ROC-AUC | 1.0000 |
| True Negatives | 88,214 |
| False Positives | 0 |
| False Negatives | 1 |
| True Positives | 1,251 |

The test set contained **1,252 fraudulent transactions**, of which **1,251 were detected**.

Because PAYSim is synthetic, these near-perfect results should be interpreted cautiously and should not be assumed to represent real-world fraud detection performance.

---

## 10. SHAP Explainability

SHAP is used to explain individual Random Forest predictions.

For each investigated transaction, the application:

1. Calculates the fraud risk score.
2. Generates SHAP values.
3. Ranks features by absolute SHAP contribution.
4. Displays the top risk drivers.
5. Converts technical feature names into investigator-friendly labels.

Example risk drivers include:

- Balance consistency features
- Amount-to-origin balance ratio
- Sender balance change
- Transaction type
- Sender balance depletion

Important:

> SHAP values explain model behaviour. They are not independent proof that a transaction is fraudulent.

---

## 11. Investigation Evidence

The application provides transaction-level evidence such as:

- Sender balance depletion
- Amount-to-origin balance ratio
- Sender transaction history
- Receiver transaction history
- Destination balance
- Transaction timing

This evidence is presented separately from the model prediction so that an investigator can review the underlying transaction behaviour.

---

## 12. RAG-Based Investigation Guidance

A lightweight local knowledge base provides investigation guidance.

Current knowledge-base entries include:

```text
KB001 - Balance Depletion
KB002 - Amount to Balance Ratio
KB003 - New Receiver Behaviour
KB004 - Historical Behaviour
KB005 - Model Risk Score
KB006 - SHAP Explanation
KB007 - Human Decision
```

The retrieval layer connects relevant investigation guidance to the transaction's risk signals.

The RAG component is used as an **investigation assistant**, not as the fraud classifier.

---

## 13. Human-in-the-Loop Decision

The final workflow does not automatically declare a transaction legally fraudulent.

Instead:

```text
ML Prediction
     +
SHAP Explanation
     +
Transaction Evidence
     +
Investigation Guidance
     ↓
Human Investigator
     ↓
Final Decision
```

The investigator can select:

- **Approve / Legitimate**
- **Escalate / Suspicious**
- **Request More Evidence**

Investigator notes are also stored with the investigation case.

This makes the application closer to a practical fraud-investigation workflow rather than a simple binary classifier.

---

## 14. Example Investigation

Example transaction:

```text
Transaction Type: TRANSFER
Amount: ₹480,404.03
Step: 632
Risk Score: 99.98%
Prediction: Fraud
```

Important transaction signals:

```text
Sender balance depleted: Yes
Amount-to-origin ratio: 1.00
Sender transaction history: 0
Receiver transaction history: 0
Destination balance: ₹0.00
Timing: Weekend
```

The application recommends escalation for human investigation.

The example was manually recorded as:

```text
Decision: Escalate / Suspicious

Notes:
High-risk transfer with complete sender balance depletion
and a new receiver. Escalated for human review.
```

---

## 15. Important Interpretation

The system deliberately separates:

### Prediction

What the ML model predicts.

### Explanation

Why the model's prediction was influenced by particular features.

### Evidence

What is observable from the transaction and historical behaviour.

### Investigation Guidance

What the knowledge base recommends considering.

### Human Decision

The final investigator action.

This separation reduces the risk of treating an automated prediction as an unquestionable fraud decision.

---

## 16. Flask Deployment

The project is deployed as a Flask web application.

The application loads:

```text
fraud_detection_model.pkl
fraud_detection_scaler.pkl
fraud_detection_feature_columns.pkl
fraud_detection_model_metadata.pkl
final_fraud_investigation_case.pkl
```

Main routes include:

```text
/                  → Transaction input page
/predict           → ML prediction and investigation result
/human-decision    → Save investigator decision
```

The application runs locally using Flask.

---

## 17. Project Structure

```text
Fraud_detection/
│
├── app.py
├── requirements.txt
├── README.md
│
├── fraud_detection_model.pkl
├── fraud_detection_scaler.pkl
├── fraud_detection_feature_columns.pkl
├── fraud_detection_model_metadata.pkl
├── final_fraud_investigation_case.pkl
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── static/
│   └── style.css
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Historical_Features.ipynb
│   └── 04_ML_Preprocessing.ipynb
│
└── data/
    ├── paysim.csv
    ├── train_dataset.parquet
    ├── validation_dataset.parquet
    └── test_dataset.parquet
```

---

## 18. Installation

Create and activate the virtual environment:

```bash
python -m venv venv
```

Windows PowerShell:

```bash
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The project uses the same Scikit-learn version used when the model artifact was created.

---

## 19. Run the Application

From the project directory:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

Enter transaction information and submit it for fraud risk assessment.

---

## 20. Technologies Used

### Programming

- Python

### Data Science

- Pandas
- NumPy
- Scikit-learn

### Machine Learning

- Random Forest
- StandardScaler

### Explainable AI

- SHAP

### Retrieval / Investigation

- Local knowledge-base retrieval
- RAG-style evidence generation

### Deployment

- Flask
- HTML
- CSS

### Serialization

- Joblib

---

## 21. Limitations

### Synthetic data

PAYSim is synthetic and does not fully represent real banking fraud.

### Extremely high test performance

The near-perfect metrics indicate that the synthetic problem may be easier to separate than real-world fraud.

### RAG scope

The current RAG implementation is a lightweight local evidence-retrieval system rather than a production-scale vector database and semantic retrieval architecture.

### Human review

The system supports investigation prioritization. It does not replace institutional fraud policies, compliance procedures, or human investigation.

---

## 22. Future Improvements

Potential future improvements include:

- Real-world transaction data
- Cost-sensitive learning
- Threshold optimization based on business costs
- Model calibration
- Time-window behavioural features
- Online/streaming fraud scoring
- Vector-database semantic retrieval
- LLM-based investigation summaries
- Case management and audit logging
- Role-based investigator access
- Production database persistence
- Model monitoring and drift detection

---

## 23. Final Architecture

```text
                 ┌──────────────────────┐
                 │   Transaction Input  │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Feature Engineering  │
                 │ + Historical Signals │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │   Random Forest ML   │
                 └──────────┬───────────┘
                            ↓
                ┌───────────┴───────────┐
                ↓                       ↓
          Fraud Prediction         Risk Score
                │                       │
                └───────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │   SHAP Explanation   │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Transaction Evidence │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ RAG Investigation    │
                 │ Guidance             │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Investigation        │
                 │ Recommendation       │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Human Investigator   │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Final Decision +     │
                 │ Investigator Notes   │
                 └──────────────────────┘
```

---

## 24. Project Objective

The final objective of this project is to demonstrate an end-to-end **Fraud Detection and Transaction Risk Scoring System** that combines:

**Machine Learning + Explainable AI + Retrieval-Augmented Investigation + Human-in-the-Loop Decision Making.**
