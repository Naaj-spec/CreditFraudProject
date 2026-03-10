# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Split into train/test
X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train SVM
svm_model = SVC(kernel='rbf', class_weight='balanced', probability=True)
svm_model.fit(X_train_scaled, y_train)
print("Training completed")

# Save model & scaler
joblib.dump(svm_model, "model/svm_fraud_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
print("Model and scaler saved")

# Save test CSV for demo
X_test.to_csv("data/test_transactions.csv", index=False)
y_test.to_csv("data/test_labels.csv", index=False)
print("Test CSV saved")
