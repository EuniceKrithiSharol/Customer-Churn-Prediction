import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


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


def prepare_data(df):

    model_data = pd.get_dummies(
        df,
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

    return X, y


def train_churn_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

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

    metrics = {

        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_test,
            predictions,
            zero_division=0
        )
    }

    return (
        model,
        scaler,
        metrics
    )


if __name__ == "__main__":

    data = generate_customer_data()

    X, y = prepare_data(
        data
    )

    model, scaler, metrics = train_churn_model(
        X,
        y
    )

    print(
        "Customer Churn Model Training Completed"
    )

    print()

    for metric, value in metrics.items():

        print(
            f"{metric}: {value:.4f}"
        )
