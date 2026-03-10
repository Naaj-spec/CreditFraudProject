import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import numpy as np

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("Machine Learning–Based Credit Card Fraud Detection System")
st.caption("Support Vector Machine (SVM) | Fraud Risk Prediction & Evaluation")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model/svm_fraud_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

model, scaler = load_model()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("⚙️ Detection Settings")

threshold = st.sidebar.slider(
    "Fraud Risk Threshold",
    0.0, 1.0, 0.5, 0.05
)

# Increased transaction processing options
max_rows = st.sidebar.selectbox(
    "Rows to Process",
    [2000, 5000, 10000, 20000, 30000, 50000],
    index=3
)

# --------------------------------------------------
# FAST CSV LOADER
# --------------------------------------------------
@st.cache_data
def load_data(file, rows):
    return pd.read_csv(file, nrows=rows, low_memory=False)

uploaded_file = st.file_uploader(
    "Upload transaction dataset (CSV format)",
    type="csv"
)

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if uploaded_file:

    with st.spinner("Processing transactions and running fraud detection..."):
        data = load_data(uploaded_file, max_rows)

    st.success(f"Loaded {len(data):,} rows for fraud detection")

    has_ground_truth = "Class" in data.columns

    if has_ground_truth:
        y_true = data["Class"]

    # Drop label
    X = data.drop(columns=["Class"], errors="ignore")

    # Scale
    X_scaled = scaler.transform(X)

    # Predict
    fraud_prob = model.predict_proba(X_scaled)[:, 1]
    fraud_pred = (fraud_prob >= threshold).astype(int)

    data["Fraud_Prediction"] = fraud_pred
    data["RiskScore"] = fraud_prob

    total = len(data)
    fraud_count = int(np.sum(fraud_pred))
    legit_count = total - fraud_count

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------
    st.subheader("Prediction Summary")

    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions Processed", f"{total:,}")
    c2.metric("Fraud Detected", f"{fraud_count:,}")
    c3.metric("Legitimate", f"{legit_count:,}")

    # --------------------------------------------------
    # TABS
    # --------------------------------------------------
    tab1, tab2, tab3 = st.tabs([
        "Transactions",
        "Fraud Only",
        "Model Evaluation"
    ])

    # --------------------------------------------------
    # TAB 1 – TRANSACTIONS
    # --------------------------------------------------
    with tab1:
        st.subheader("Transaction Preview")

        st.dataframe(data.head(1000), use_container_width=True)
        st.info("Showing first 1000 rows for performance.")

    # --------------------------------------------------
    # TAB 2 – FRAUD ONLY
    # --------------------------------------------------
    with tab2:

        fraud_data = data[data["Fraud_Prediction"] == 1]

        if fraud_data.empty:
            st.success("No fraud detected.")
        else:
            st.dataframe(fraud_data.head(1000), use_container_width=True)

            csv = fraud_data.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Fraud Transactions",
                csv,
                "fraud_transactions.csv",
                "text/csv"
            )

    # --------------------------------------------------
    # TAB 3 – MODEL EVALUATION
    # --------------------------------------------------
    with tab3:

        col1, col2 = st.columns(2)

        # Fraud distribution
        with col1:
            fig1, ax1 = plt.subplots()
            ax1.pie(
                [legit_count, fraud_count],
                labels=["Legitimate", "Fraud"],
                autopct="%1.1f%%"
            )
            ax1.set_title("Fraud vs Legitimate Distribution")
            st.pyplot(fig1)

        # Risk score distribution
        with col2:
            fig2, ax2 = plt.subplots()
            ax2.hist(data["RiskScore"], bins=20)
            ax2.axvline(threshold, linestyle="--")
            ax2.set_title("Fraud Risk Score Distribution")
            st.pyplot(fig2)

        st.divider()

        if has_ground_truth:

            # -------------------------
            # CONFUSION MATRIX
            # -------------------------
            cm = confusion_matrix(y_true, fraud_pred)

            fig_cm, ax_cm = plt.subplots()
            ax_cm.imshow(cm)
            ax_cm.set_title("Confusion Matrix")

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax_cm.text(j, i, cm[i, j], ha="center")

            st.pyplot(fig_cm)

            # -------------------------
            # PERFORMANCE METRICS
            # -------------------------
            accuracy = accuracy_score(y_true, fraud_pred) * 100
            precision = precision_score(y_true, fraud_pred, zero_division=0) * 100
            recall = recall_score(y_true, fraud_pred, zero_division=0) * 100
            f1 = f1_score(y_true, fraud_pred, zero_division=0) * 100

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Accuracy", f"{accuracy:.2f}%")
            c2.metric("Precision", f"{precision:.2f}%")
            c3.metric("Recall", f"{recall:.2f}%")
            c4.metric("F1 Score", f"{f1:.2f}%")

            st.divider()

            # -------------------------
            # ROC CURVE + AUC
            # -------------------------
            st.subheader("ROC Curve and AUC Score")

            fpr, tpr, _ = roc_curve(y_true, fraud_prob)
            roc_auc = auc(fpr, tpr)

            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            ax_roc.plot([0, 1], [0, 1], linestyle="--")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("Receiver Operating Characteristic (ROC)")
            ax_roc.legend()

            st.pyplot(fig_roc)

            st.metric("ROC-AUC Score", f"{roc_auc*100:.2f}%")

        else:
            st.warning("Dataset has no 'Class' column. Evaluation unavailable.")

else:
    st.info("Upload a CSV dataset to start.")