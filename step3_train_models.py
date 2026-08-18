"""Step 3: train five classification models and compare metrics."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


DATA_PATH = Path(__file__).resolve().parent / "cleaned_credit_data.csv"
MODEL_DIR = Path(__file__).resolve().parent / "model"
TARGET_COLUMN = "default payment next month"


def build_preprocessor(X):
    """Apply scaling to numeric variables and one-hot encoding to categorical variables."""
    numeric_features = [
        "LIMIT_BAL",
        "AGE",
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
    ]
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

    # Numeric features are standardized because their scales differ widely across bills,
    # limits, and payment amounts. Scaling prevents features with large magnitudes from
    # dominating the optimization of the classifier.
    numeric_transformer = StandardScaler()

    # Categorical variables are encoded using one-hot encoding so that values such as
    # sex and education are treated as distinct categories rather than arbitrary numbers.
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor


def evaluate_model(model, X_test, y_test):
    """Return the six classification metrics required by the assignment."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC Score": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def train_and_evaluate_models():
    MODEL_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        "KNN Classifier": KNeighborsClassifier(n_neighbors=5),
        "Gaussian Naive Bayes": GaussianNB(),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            min_samples_leaf=10,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }

    results = []

    for model_name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("model", estimator),
            ]
        )

        pipeline.fit(X_train, y_train)

        model_path = MODEL_DIR / f"{model_name.lower().replace(' ', '_')}.joblib"
        joblib.dump(pipeline, model_path, compress=9)

        metrics = evaluate_model(pipeline, X_test, y_test)
        metrics["Model"] = model_name
        results.append(metrics)
        print(f"\n=== {model_name} ===")
        print(pd.DataFrame([metrics]).drop(columns=["Model"]).to_string(index=False))

    comparison_df = pd.DataFrame(results).set_index("Model")
    comparison_df = comparison_df[[
        "Accuracy",
        "AUC Score",
        "Precision",
        "Recall",
        "F1 Score",
        "MCC",
    ]]

    comparison_df.to_csv(Path(__file__).resolve().parent / "model_comparison.csv")
    print("\n=== Full model comparison ===")
    print(comparison_df.round(4).to_string())


if __name__ == "__main__":
    train_and_evaluate_models()
