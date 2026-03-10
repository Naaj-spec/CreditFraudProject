💳 Machine Learning–Based Credit Card Fraud Detection System
A Streamlit web application that detects fraudulent credit card transactions using a Support Vector Machine (SVM) machine learning model.
The system allows users to upload transaction datasets, run fraud detection, and visualize results through an interactive dashboard.

📌 Project Overview
Credit card fraud is a major financial risk for banks, financial institutions, and online payment systems. This project applies machine learning techniques to automatically detect suspicious transactions based on transaction features.
The application provides:
•	Fraud prediction using a trained SVM model
•	Transaction risk scoring
•	Fraud detection threshold control
•	Model performance evaluation metrics
•	Visual analytics dashboard
The interface is built using Streamlit, enabling fast and interactive data analysis.

⚙️ System Architecture
The system workflow follows these stages:
1.	Dataset upload
2.	Data preprocessing and scaling
3.	Fraud probability prediction using SVM
4.	Threshold-based classification
5.	Visualization and evaluation metrics

🚀 Features
1️⃣ Transaction Processing
•	Upload CSV transaction datasets
•	Process up to 50,000 transactions
•	Automatic feature scaling using a trained scaler

2️⃣ Fraud Prediction
•	Uses a trained Support Vector Machine (SVM) classifier
•	Generates:
o	Fraud prediction
o	Fraud probability risk score

3️⃣ Interactive Dashboard
The Streamlit interface provides:
•	Transaction preview
•	Fraud-only filtered results
•	Downloadable fraud reports

4️⃣ Model Evaluation
When the dataset includes a Class column (ground truth) the system computes:
•	Confusion Matrix
•	Accuracy
•	Precision
•	Recall
•	F1 Score
•	ROC Curve
•	ROC-AUC Score

🖥️ User Interface Tabs
Tab	Description
Transactions	Displays processed transactions
Fraud Only	Shows only detected fraud cases
Model Evaluation	Displays metrics and performance charts

📊 Example Output
The system produces the following visualizations:
•	Fraud vs Legitimate transaction distribution
•	Risk score distribution histogram
•	Confusion matrix
•	ROC curve
These visualizations help users evaluate model performance.

🧠 Machine Learning Model
The system uses:
•	Algorithm: Support Vector Machine (SVM)
•	Libraries:
o	Scikit-learn
o	Pandas
o	NumPy
The model outputs a fraud probability score for each transaction.

📂 Project Structure
fraud-detection-system/
│
├── app.py                # Streamlit application
├── requirements.txt      # Python dependencies
│
├── model/
│   ├── svm_fraud_model.pkl
│   └── scaler.pkl
│
├── datasets/
│   └── sample_transactions.csv
│
└── README.md

🛠️ Installation Guide
1️⃣ Clone the Repository
git clone https://github.com/yourusername/fraud-detection-system.git
cd fraud-detection-system

2️⃣ Create a Virtual Environment
python -m venv venv
Activate the environment:
Windows
venv\Scripts\activate
Mac/Linux
source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Application
streamlit run app.py
The application will open in your browser.

📥 Dataset Format
The system expects a CSV dataset with transaction features.
Example structure:

Feature1	Feature2	Feature3	...	Class
0.23	-1.44	0.56	...	0
-2.11	1.22	-0.78	...	1

Where:
•	0 = Legitimate transaction
•	1 = Fraudulent transaction
The Class column is optional, but required for model evaluation.

📈 Performance Metrics Explained
Metric	Description
Accuracy	Overall correctness of the model
Precision	Percentage of predicted fraud that is actually fraud
Recall	Percentage of actual fraud detected
F1 Score	Balance between precision and recall
ROC-AUC	Model ability to distinguish fraud vs legitimate

🔒 Limitations
•	The model performance depends on dataset quality.
•	Real-world fraud detection requires continuous retraining.
•	Class imbalance in fraud datasets may affect prediction accuracy.

🔮 Future Improvements
Potential enhancements include:
•	Deep learning fraud detection models
•	Real-time transaction monitoring
•	API integration for banking systems
•	Fraud alert notification systems
•	Blockchain transaction verification

👨‍💻 Author

NAJIB ABDI

Bachelor of Information Technology
Machine Learning & Software Development

📜 License
This project is released under the MIT License.

