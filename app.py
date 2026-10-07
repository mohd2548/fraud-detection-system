from flask import Flask, render_template,request
import joblib
import os
import pandas as pd 
import shap

app = Flask(__name__)

MODEL_PATH = "fraud_detection_model.pkl"
SCALER_PATH = "fraud_detection_scaler.pkl"
FEATURES_PATH = "fraud_detection_feature_columns.pkl"
METADATA_PATH = "fraud_detection_model_metadata.pkl"
CASE_PATH = "final_fraud_investigation_case.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURES_PATH)
model_metadata = joblib.load(METADATA_PATH)
explainer = shap.TreeExplainer(model)
investigation_case = joblib.load(CASE_PATH)

print("===== FRAUD DETECTION APP =====")
print("Model loaded:", type(model).__name__)
print("Scaler loaded:", type(scaler).__name__)
print("Features:", len(feature_columns))
print("Model metadata loaded:", type(model_metadata).__name__)
print("SHAP explainer loaded")

SHAP_LABELS = {
    "abs_origin_balance_error": "Balance Consistency Feature",
    "origin_balance_error": "Origin Balance Difference Feature",
    "amount_to_origin_ratio": "Amount-to-Origin Balance Ratio",
    "sender_balance_change": "Sender Balance Change",
    "type_TRANSFER": "Transfer Transaction Type",
    "type_CASH_OUT": "Cash-Out Transaction Type",
    "type_PAYMENT": "Payment Transaction Type",
    "type_CASH_IN": "Cash-In Transaction Type",
    "type_DEBIT": "Debit Transaction Type",
    "sender_balance_depleted": "Sender Balance Depleted",
    "destination_balance_depleted": "Destination Balance Depleted",
    "origin_balance_zero": "Origin Balance Was Zero",
    "sender_past_transaction_count": "Sender Transaction History",
    "sender_past_total_amount": "Sender Historical Transaction Amount",
    "sender_past_avg_amount": "Sender Historical Average Amount",
    "sender_past_unique_receivers": "Sender Unique Receivers",
    "receiver_past_transaction_count": "Receiver Transaction History",
    "receiver_past_total_amount": "Receiver Historical Transaction Amount",
    "receiver_past_avg_amount": "Receiver Historical Average Amount",
    "receiver_past_unique_senders": "Receiver Unique Senders",
    "hour_of_day": "Transaction Hour",
    "day_of_week": "Transaction Day",
    "is_weekend": "Weekend Transaction",
    "amount": "Transaction Amount",
    "oldbalanceOrg": "Previous Origin Balance",
    "newbalanceOrig": "New Origin Balance",
    "oldbalanceDest": "Previous Destination Balance",
    "newbalanceDest": "New Destination Balance",
    "receiver_balance_change": "Receiver Balance Change",
    "receiver_amount_difference": "Receiver Amount Difference",
    "step": "Transaction Step",
    "isFlaggedFraud": "System Fraud Flag"
}

SHAP_EXPLANATIONS = {
    "abs_origin_balance_error":
        "This engineered balance-consistency feature contributed to the model's prediction. For this transaction, the calculated balance error is 0.",

    "origin_balance_error":
        "This engineered balance feature contributed to the model's prediction. For this transaction, the calculated origin balance difference is 0.",
    
    "amount_to_origin_ratio":
        "The transaction amount was very high relative to the sender's available balance.",

    "sender_balance_change":
        "The sender's balance changed significantly as a result of the transaction.",

    "type_TRANSFER":
        "The transaction was a transfer, which contributed to the model's risk assessment.",

    "sender_balance_depleted":
        "The sender's balance was completely depleted after the transaction.",

    "destination_balance_depleted":
        "The destination balance was zero after the transaction.",

    "sender_past_transaction_count":
        "The sender's historical transaction activity influenced the model's assessment.",

    "receiver_past_transaction_count":
        "The receiver's historical transaction activity influenced the model's assessment.",

    "hour_of_day":
        "The transaction time contributed to the model's risk assessment.",

    "is_weekend":
        "The transaction occurred during the weekend, which influenced the model's assessment."
}

@app.route("/")
def home():
    return render_template(
        "index.html",
        model_name=type(model).__name__,
        feature_count=len(feature_columns)
    )

@app.route("/predict", methods=["POST"])
def predict():

    step = int(request.form["step"])
    transaction_type = request.form["type"]

    amount = float(request.form["amount"])

    oldbalanceOrg = float(request.form["oldbalanceOrg"])
    newbalanceOrig = float(request.form["newbalanceOrig"])

    oldbalanceDest = float(request.form["oldbalanceDest"])
    newbalanceDest = float(request.form["newbalanceDest"])

    isFlaggedFraud = int(request.form["isFlaggedFraud"])

    sender_past_transaction_count = int(
        request.form["sender_past_transaction_count"]
    )

    sender_past_total_amount = float(
        request.form["sender_past_total_amount"]
    )

    sender_past_avg_amount = float(
        request.form["sender_past_avg_amount"]
    )

    sender_past_unique_receivers = int(
        request.form["sender_past_unique_receivers"]
    )

    receiver_past_transaction_count = int(
        request.form["receiver_past_transaction_count"]
    )

    receiver_past_total_amount = float(
        request.form["receiver_past_total_amount"]
    )

    receiver_past_avg_amount = float(
        request.form["receiver_past_avg_amount"]
    )

    receiver_past_unique_senders = int(
        request.form["receiver_past_unique_senders"]
    )



    origin_balance_error = (
        oldbalanceOrg - amount - newbalanceOrig
    )

    abs_origin_balance_error = abs(
        origin_balance_error
    )

    if oldbalanceOrg > 0:
        amount_to_origin_ratio = (
            amount / oldbalanceOrg
        )
    else:
        amount_to_origin_ratio = 0.0

    sender_balance_depleted = int(
        newbalanceOrig == 0
    )

    destination_balance_depleted = int(
        newbalanceDest == 0
    )

    origin_balance_zero = int(
        oldbalanceOrg == 0
    )

    hour_of_day = step % 24

    day_of_week = (step // 24) % 7

    is_weekend = int(
        day_of_week >= 5
    )

    sender_balance_change = (
        oldbalanceOrg - newbalanceOrig
    )

    receiver_balance_change = (
        newbalanceDest - oldbalanceDest
    )

    receiver_amount_difference = (
        amount - receiver_balance_change
    )

    type_CASH_IN = int(
        transaction_type == "CASH_IN"
    )

    type_CASH_OUT = int(
        transaction_type == "CASH_OUT"
    )

    type_DEBIT = int(
        transaction_type == "DEBIT"
    )

    type_PAYMENT = int(
        transaction_type == "PAYMENT"
    )

    type_TRANSFER = int(
        transaction_type == "TRANSFER"
    )


    input_data = {

        "step": step,
        "amount": amount,

        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,

        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,

        "isFlaggedFraud": isFlaggedFraud,

        "origin_balance_error": origin_balance_error,
        "abs_origin_balance_error": abs_origin_balance_error,

        "amount_to_origin_ratio": amount_to_origin_ratio,

        "sender_balance_depleted": sender_balance_depleted,
        "destination_balance_depleted": destination_balance_depleted,
        "origin_balance_zero": origin_balance_zero,

        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,

        "sender_balance_change": sender_balance_change,
        "receiver_balance_change": receiver_balance_change,
        "receiver_amount_difference": receiver_amount_difference,

        "type_CASH_IN": type_CASH_IN,
        "type_CASH_OUT": type_CASH_OUT,
        "type_DEBIT": type_DEBIT,
        "type_PAYMENT": type_PAYMENT,
        "type_TRANSFER": type_TRANSFER,

        "receiver_past_transaction_count":
            receiver_past_transaction_count,

        "receiver_past_total_amount":
            receiver_past_total_amount,

        "receiver_past_avg_amount":
            receiver_past_avg_amount,

        "receiver_past_unique_senders":
            receiver_past_unique_senders,

        "sender_past_transaction_count":
            sender_past_transaction_count,

        "sender_past_total_amount":
            sender_past_total_amount,

        "sender_past_avg_amount":
            sender_past_avg_amount,

        "sender_past_unique_receivers":
            sender_past_unique_receivers
    }


    X = pd.DataFrame([input_data])

    X = X[feature_columns]
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]
    risk_score = model.predict_proba(X_scaled)[0][1]

    shap_values = explainer.shap_values(X_scaled)

    if isinstance(shap_values, list):
        shap_values = shap_values[1][0]

    elif len(shap_values.shape) == 3:
        shap_values = shap_values[0, :, 1]

    else:
        shap_values = shap_values[0]

    top_features = (
        pd.DataFrame({
            "feature": feature_columns,
            "shap_value": shap_values
        })
        .assign(abs_shap=lambda df: df["shap_value"].abs())
        .sort_values("abs_shap", ascending=False)
        .head(5)
    )

    top_features = top_features.to_dict(orient="records")
    
    for item in top_features:

        item["label"] = SHAP_LABELS.get(
            item["feature"],
            item["feature"]
        )

        item["explanation"] = SHAP_EXPLANATIONS.get(
            item["feature"],
            "This feature influenced the model's risk assessment."
        )
    
    transaction_evidence = {
        "sender_balance_depleted": sender_balance_depleted,
        "amount_to_origin_ratio": amount_to_origin_ratio,
        "sender_past_transaction_count": sender_past_transaction_count,
        "receiver_past_transaction_count": receiver_past_transaction_count,
        "new_balance_destination": newbalanceDest,
        "is_weekend": is_weekend
    }


    investigation_case["current_transaction_evidence"] = transaction_evidence
    investigation_case["current_prediction"] = {
        "prediction": int(prediction),
        "risk_score": float(risk_score),
        "transaction_type": transaction_type,
        "amount": amount,
        "step": step,
        "top_features": top_features
    }
    
    return render_template(
        "result.html",
        prediction=int(prediction),
        risk_score=float(risk_score),
        transaction_type=transaction_type,
        amount=amount,
        step=step,
        model_name=type(model).__name__,
        feature_count=len(feature_columns),
        top_features=top_features,
        investigation_case=investigation_case,
        transaction_evidence=transaction_evidence    
    )


@app.route("/human-decision", methods=["POST"])
def human_decision():

    decision = request.form["decision"]
    investigator_notes = request.form["investigator_notes"]

    investigation_case["case"]["human_decision"]["decision"] = decision
    investigation_case["case"]["human_decision"]["investigator_notes"] = investigator_notes

    joblib.dump(
        investigation_case,
        CASE_PATH
    )

    current_prediction = investigation_case.get(
        "current_prediction",
        {}
    )

    current_transaction_evidence = investigation_case.get(
        "current_transaction_evidence",
        {}
    )

    return render_template(
        "result.html",
        prediction=current_prediction.get("prediction", 0),
        risk_score=current_prediction.get("risk_score", 0),
        transaction_type=current_prediction.get("transaction_type", ""),
        amount=current_prediction.get("amount", 0),
        step=current_prediction.get("step", 0),
        model_name=type(model).__name__,
        feature_count=len(feature_columns),
        top_features=current_prediction.get("top_features", []),
        investigation_case=investigation_case,
        transaction_evidence=current_transaction_evidence
    )
    
if __name__ == "__main__":
    app.run(debug=True)