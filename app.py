import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

# -------------------------------------------------
# TITLE
# -------------------------------------------------

st.title("📉 Customer Churn Prediction")

st.markdown(
    "Machine Learning based customer churn prediction "
    "using Random Forest Classification."
)

# -------------------------------------------------
# GENERATE DATA
# -------------------------------------------------

@st.cache_data
def generate_customer_data():

    np.random.seed(42)

    n = 2000

    df = pd.DataFrame({

        "Customer_ID": range(1, n + 1),

        "Age": np.random.randint(
            18,
            75,
            n
        ),

        "Tenure": np.random.randint(
            1,
            120,
            n
        ),

        "Monthly_Charges": np.random.randint(
            20,
            150,
            n
        ),

        "Total_Spend": np.random.randint(
            500,
            15000,
            n
        ),

        "Support_Calls": np.random.randint(
            0,
            15,
            n
        ),

        "Contract_Type": np.random.choice(
            [
                "Monthly",
                "One Year",
                "Two Year"
            ],
            n,
            p=[0.5, 0.3, 0.2]
        ),

        "Payment_Method": np.random.choice(
            [
                "Credit Card",
                "Bank Transfer",
                "Electronic Check",
                "Cash"
            ],
            n
        )
    })

    churn_probability = (
        0.15
        +
        (df["Monthly_Charges"] / 1000)
        +
        (df["Support_Calls"] / 40)
        -
        (df["Tenure"] / 500)
    )

    churn_probability = np.clip(
        churn_probability,
        0.05,
        0.90
    )

    df["Churn"] = np.random.binomial(
        1,
        churn_probability
    )

    return df


df = generate_customer_data()

# -------------------------------------------------
# DATA PREPROCESSING
# -------------------------------------------------

model_data = df.copy()

model_data = pd.get_dummies(
    model_data,
    columns=[
        "Contract_Type",
        "Payment_Method"
    ],
    drop_first=True
)

X = model_data.drop(
    columns=[
        "Customer_ID",
        "Churn"
    ]
)

y = model_data["Churn"]

# -------------------------------------------------
# TRAIN TEST SPLIT
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------------------------
# FEATURE SCALING
# -------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

# -------------------------------------------------
# MODEL TRAINING
# -------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(
    X_train_scaled,
    y_train
)

predictions = model.predict(
    X_test_scaled
)

# -------------------------------------------------
# MODEL PERFORMANCE
# -------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

# -------------------------------------------------
# KPI METRICS
# -------------------------------------------------

st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Accuracy",
    f"{accuracy:.2%}"
)

col2.metric(
    "Precision",
    f"{precision:.2%}"
)

col3.metric(
    "Recall",
    f"{recall:.2%}"
)

col4.metric(
    "F1 Score",
    f"{f1:.2%}"
)

st.divider()

# -------------------------------------------------
# CONFUSION MATRIX
# -------------------------------------------------

st.subheader("🔍 Confusion Matrix")

cm = confusion_matrix(
    y_test,
    predictions
)

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual No Churn",
        "Actual Churn"
    ],
    columns=[
        "Predicted No Churn",
        "Predicted Churn"
    ]
)

st.dataframe(
    cm_df,
    use_container_width=True
)

# -------------------------------------------------
# FEATURE IMPORTANCE
# -------------------------------------------------

st.subheader("📈 Feature Importance")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

fig_importance = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Factors Influencing Customer Churn"
)

st.plotly_chart(
    fig_importance,
    use_container_width=True
)

# -------------------------------------------------
# CHURN DISTRIBUTION
# -------------------------------------------------

st.subheader("👥 Customer Churn Distribution")

churn_counts = (
    df["Churn"]
    .value_counts()
    .reset_index()
)

churn_counts.columns = [
    "Churn",
    "Customers"
]

churn_counts["Churn"] = churn_counts[
    "Churn"
].map({
    0: "No Churn",
    1: "Churn"
})

fig_churn = px.pie(
    churn_counts,
    names="Churn",
    values="Customers",
    title="Customer Churn Distribution"
)

st.plotly_chart(
    fig_churn,
    use_container_width=True
)

# -------------------------------------------------
# CUSTOMER CHURN PREDICTOR
# -------------------------------------------------

st.divider()

st.subheader("🤖 Predict Customer Churn")

st.markdown(
    "Enter customer information to estimate "
    "the probability of customer churn."
)

col1, col2 = st.columns(2)

with col1:

    age = st.slider(
        "Age",
        18,
        75,
        35
    )

    tenure = st.slider(
        "Tenure (Months)",
        1,
        120,
        24
    )

    monthly_charges = st.slider(
        "Monthly Charges",
        20,
        150,
        75
    )

with col2:

    total_spend = st.number_input(
        "Total Spend",
        min_value=500,
        max_value=15000,
        value=3000
    )

    support_calls = st.slider(
        "Support Calls",
        0,
        15,
        2
    )

    contract_type = st.selectbox(
        "Contract Type",
        [
            "Monthly",
            "One Year",
            "Two Year"
        ]
    )

payment_method = st.selectbox(
    "Payment Method",
    [
        "Credit Card",
        "Bank Transfer",
        "Electronic Check",
        "Cash"
    ]
)

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------

if st.button("Predict Churn"):

    input_data = pd.DataFrame({
        "Age": [age],
        "Tenure": [tenure],
        "Monthly_Charges": [monthly_charges],
        "Total_Spend": [total_spend],
        "Support_Calls": [support_calls],
        "Contract_Type": [contract_type],
        "Payment_Method": [payment_method]
    })

    input_data = pd.get_dummies(
        input_data
    )

    input_data = input_data.reindex(
        columns=[
            column
            for column in model_data.columns
            if column not in [
                "Customer_ID",
                "Churn"
            ]
        ],
        fill_value=0
    )

    input_scaled = scaler.transform(
        input_data
    )

    prediction = model.predict(
        input_scaled
    )[0]

    probability = model.predict_proba(
        input_scaled
    )[0][1]

    if prediction == 1:

        st.error(
            f"⚠️ High Churn Risk: "
            f"{probability:.2%}"
        )

    else:

        st.success(
            f"✅ Low Churn Risk: "
            f"{probability:.2%}"
        )

# -------------------------------------------------
# DATASET PREVIEW
# -------------------------------------------------

st.divider()

st.subheader("📄 Customer Dataset Preview")

st.dataframe(
    df.head(100),
    use_container_width=True
)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.caption(
    "Customer Churn Prediction | "
    "Python • Scikit-learn • Random Forest • Streamlit"
)
