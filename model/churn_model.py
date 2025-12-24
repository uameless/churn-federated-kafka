from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COL = "Churn"


@dataclass
class DatasetBundle:
    df: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    categorical_cols: List[str]
    numeric_cols: List[str]


def load_telco_csv(csv_path: str) -> pd.DataFrame:
    """
    Loads Telco Customer Churn CSV and performs minimal cleaning:
    - Ensures TotalCharges is numeric, coerces errors to NaN, imputes later.
    - Maps target Churn: Yes/No -> 1/0
    """
    df = pd.read_csv(csv_path)

    # Standard Telco dataset: TotalCharges sometimes has blank strings.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Target: Yes/No
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column '{TARGET_COL}' in CSV.")

    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0}).astype(int)

    # Drop customerID if present (identifier)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    return df


def infer_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Infers categorical and numeric columns.
    """
    feature_df = df.drop(columns=[TARGET_COL])
    categorical_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = feature_df.select_dtypes(include=[np.number, "bool"]).columns.tolist()

    # Ensure no overlaps
    categorical_cols = [c for c in categorical_cols if c not in numeric_cols]
    return categorical_cols, numeric_cols


def build_preprocessor(categorical_cols: List[str], numeric_cols: List[str]) -> ColumnTransformer:
    """
    Preprocessing:
    - Categorical: impute most_frequent + OneHotEncoder(handle_unknown='ignore')
    - Numeric: impute median + StandardScaler
    """
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, categorical_cols),
            ("num", num_pipe, numeric_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def build_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    """
    Logistic Regression classifier (scikit-learn).
    """
    clf = LogisticRegression(max_iter=300, solver="lbfgs")
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])
    return pipe


def prepare_dataset(csv_path: str, test_size: float = 0.2, random_state: int = 42) -> DatasetBundle:
    df = load_telco_csv(csv_path)
    categorical_cols, numeric_cols = infer_feature_columns(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return DatasetBundle(
        df=df,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
    )


def evaluate_binary_classifier(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return {"accuracy": float(acc), "roc_auc": float(auc), "fpr": fpr, "tpr": tpr}


def extract_lr_weights_from_pipeline(pipe: Pipeline) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (coef, intercept) from a fitted pipeline with final step 'clf' = LogisticRegression.
    coef shape: (1, n_features)
    intercept shape: (1,)
    """
    clf: LogisticRegression = pipe.named_steps["clf"]
    return clf.coef_.copy(), clf.intercept_.copy()


def create_lr_from_weights(coef: np.ndarray, intercept: np.ndarray) -> LogisticRegression:
    """
    Reconstruct a fitted LogisticRegression from FedAvg weights.
    This is REQUIRED for sklearn to allow predict / predict_proba.
    """
    lr = LogisticRegression()

    # Required sklearn fitted attributes
    lr.classes_ = np.array([0, 1], dtype=int)
    lr.coef_ = coef.astype(float)
    lr.intercept_ = intercept.astype(float)
    lr.n_features_in_ = coef.shape[1]

    return lr
