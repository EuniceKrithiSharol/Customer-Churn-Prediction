# 📉 Customer Churn Prediction

A Machine Learning project that predicts whether customers are likely to leave a company using customer behavior data and supervised classification algorithms.

---

## 🚀 Project Overview

Customer churn is one of the most important business problems for subscription-based and service-oriented companies.

This project uses Machine Learning to identify customers who are at risk of leaving.

The model analyzes customer characteristics and behavior to estimate churn probability and provide insights into factors that influence customer retention.

---

## 🧠 Machine Learning Problem

### Problem Type

Binary Classification.

The model predicts:

```text
0 → Customer Will Not Churn
1 → Customer Will Churn
```

---

## 🤖 Machine Learning Model

### Random Forest Classifier

Random Forest is an ensemble machine learning algorithm that combines multiple decision trees to improve prediction performance.

The model is trained using customer features including:

- Age
- Customer Tenure
- Monthly Charges
- Total Spend
- Support Calls
- Contract Type
- Payment Method

---

## ✨ Features

- Interactive churn prediction dashboard
- Customer churn probability prediction
- Random Forest Classification
- Feature preprocessing
- Categorical feature encoding
- Feature scaling
- Model performance evaluation
- Accuracy analysis
- Precision analysis
- Recall analysis
- F1-score analysis
- Confusion Matrix
- Feature importance visualization
- Customer churn distribution

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit
- Matplotlib
- Seaborn

---

## 📁 Project Structure

```text
Customer-Churn-Prediction/
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── app.py
│
├── data/
│   └── README.md
│
├── src/
│   └── churn_model.py
│
├── models/
│   └── README.md
│
├── notebooks/
│   └── README.md
│
└── reports/
    └── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Customer-Churn-Prediction.git
```

Move into the project directory:

```bash
cd Customer-Churn-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 📊 Dataset

The project currently uses a simulated customer dataset.

The dataset contains:

| Feature | Description |
|---|---|
| Customer ID | Unique customer identifier |
| Age | Customer age |
| Tenure | Number of months with the company |
| Monthly Charges | Monthly service cost |
| Total Spend | Total customer spending |
| Support Calls | Number of customer support interactions |
| Contract Type | Customer subscription contract |
| Payment Method | Customer payment method |
| Churn | Target variable |

---

## 🔄 Machine Learning Workflow

```text
Customer Data
      ↓
Data Preprocessing
      ↓
Feature Encoding
      ↓
Feature Scaling
      ↓
Train/Test Split
      ↓
Random Forest Model
      ↓
Model Evaluation
      ↓
Customer Churn Prediction
```

---

## 📊 Model Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 💡 Business Applications

Customer churn prediction can help businesses:

- Identify high-risk customers
- Improve customer retention
- Create targeted retention campaigns
- Reduce customer loss
- Improve customer experience
- Understand churn drivers
- Optimize customer support strategies

---

## 🔮 Future Improvements

- Logistic Regression comparison
- XGBoost model
- Hyperparameter tuning
- Cross-validation
- Real-world customer datasets
- Automated model training
- Model persistence using Joblib
- Customer risk scoring
- Retention recommendations
- Cloud deployment

---

## 👩‍💻 Author

Developed as part of a Machine Learning and Data Analytics portfolio.
