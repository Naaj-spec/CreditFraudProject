# evaluate_model.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    RocCurveDisplay
)

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="SVM Credit Card Fraud Evaluation",
    page_icon="",
    layout="wide"
)

st.title(" SVM Credit Card Fraud Detection Evaluation")
st.caption("BSc. IT | Fraud Detection Demo")

# -------------------------------
# Load model & scaler
# -------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model/svm_fraud_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

model, scaler = load_model()

# -------------------------------
# Load test data
# -------------------------------
st.sidebar.header(" Settings")

try:
    test_file = "data/test_transactions.csv"
    data = pd.read_csv(test_file)
    st.success(f"Test dataset loaded: {data.shape[0]:,} rows")
except Exception as e:
    st.error(f"Could not load test data: {e}")
    st.stop()

# -------------------------------
# Preprocessing
# -------------------------------
# If labels exist
has_labels = "Class" in data.columns

X_test = data.drop(columns=["Class"]) if has_labels else data.copy()

# Handle categorical features if any
categorical_cols = X_test.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    X_test[col] = X_test[col].fillna("NA")
    X_test[col] = pd.factorize(X_test[col])[0]

# Fill missing numeric values
X_test = X_test.fillna(0)

# Scale features
X_scaled = scaler.transform(X_test)

# -------------------------------
# Predictions
# -------------------------------
fraud_prob = model.predict_proba(X_scaled)[:, 1]
fraud_pred = (fraud_prob >= 0.5).astype(int)

data["Fraud"] = fraud_pred
data["RiskScore"] = fraud_prob

# -------------------------------
# Summary metrics
# -------------------------------
total = len(data)
fraud_count = int((data["Fraud"] == 1).sum())
legit_count = total - fraud_count

st.subheader(" Prediction Summary")
c1, c2, c3 = st.columns(3)
c1.metric("Total Transactions", f"{total:,}")
c2.metric("Fraudulent Predicted", f"{fraud_count:,}")
c3.metric("Legitimate Predicted", f"{legit_count:,}")

# -------------------------------
# Evaluation metrics if labels exist
# -------------------------------
if has_labels:
    y_true = data["Class"]
    y_pred = data["Fraud"]

    st.subheader(" Model Evaluation Metrics")
    st.text(classification_report(y_true, y_pred, digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    st.markdown("**Confusion Matrix**")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    # ROC-AUC
    auc = roc_auc_score(y_true, fraud_prob)
    st.metric("ROC-AUC Score", f"{auc:.4f}")

    fig_roc, ax_roc = plt.subplots()
    RocCurveDisplay.from_predictions(y_true, fraud_prob, ax=ax_roc)
    st.pyplot(fig_roc)

# -------------------------------
# Download predictions
# -------------------------------
csv = data.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Predictions CSV",
    csv,
    "predictions.csv",
    "text/csv"
)

st.info("Predictions ready. Visualizations and metrics displayed above.")
