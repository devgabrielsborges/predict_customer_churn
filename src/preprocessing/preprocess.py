import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    collapse_cols = [
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    for col in collapse_cols:
        if col in df.columns:
            df[col] = df[col].replace(
                {"No internet service": "No", "No phone service": "No"}
            )

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    present = [c for c in service_cols if c in df.columns]
    df["num_services"] = df[present].apply(
        lambda r: (r == "Yes").sum() + (r == "DSL").sum() + (r == "Fiber optic").sum(),
        axis=1,
    )

    addon_cols = [c for c in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"] if c in df.columns]
    df["num_addons"] = df[addon_cols].apply(lambda r: (r == "Yes").sum(), axis=1)

    streaming_cols = [c for c in ["StreamingTV", "StreamingMovies"] if c in df.columns]
    df["num_streaming"] = df[streaming_cols].apply(lambda r: (r == "Yes").sum(), axis=1)

    df["avg_charge_per_month"] = df["TotalCharges"] / df["tenure"].replace(0, np.nan)
    df["avg_charge_per_month"] = df["avg_charge_per_month"].fillna(df["MonthlyCharges"])

    df["charge_per_service"] = df["MonthlyCharges"] / df["num_services"].replace(0, np.nan)
    df["charge_per_service"] = df["charge_per_service"].fillna(0)

    df["log_tenure"] = np.log1p(df["tenure"])
    df["log_total_charges"] = np.log1p(df["TotalCharges"])

    df["tenure_bin"] = pd.cut(
        df["tenure"],
        bins=[0, 6, 12, 24, 48, 72, np.inf],
        labels=["0-6m", "6-12m", "1-2y", "2-4y", "4-6y", "6y+"],
    ).astype(str)

    df["monthly_to_total_ratio"] = df["MonthlyCharges"] / df["TotalCharges"].replace(0, np.nan)
    df["monthly_to_total_ratio"] = df["monthly_to_total_ratio"].fillna(1)

    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)
    auto_methods = ["Bank transfer (automatic)", "Credit card (automatic)"]
    df["auto_pay"] = df["PaymentMethod"].isin(auto_methods).astype(int)
    df["has_internet"] = (df["InternetService"] != "No").astype(int)
    df["has_fiber"] = (df["InternetService"] == "Fiber optic").astype(int)
    df["partner_and_dependents"] = (
        (df["Partner"] == "Yes") & (df["Dependents"] == "Yes")
    ).astype(int)
    df["alone_senior"] = (
        (df["SeniorCitizen"] == 1) & (df["Partner"] == "No") & (df["Dependents"] == "No")
    ).astype(int)
    df["month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    df["high_monthly"] = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)
    df["new_and_month_to_month"] = (df["is_new_customer"] & df["month_to_month"]).astype(int)

    return df


def preprocess_data(data: pd.DataFrame, target_column: str | None = None):
    target_column = target_column or os.getenv("TARGET_COLUMN")
    id_column = os.getenv("ID_COLUMN", "id")

    data = _engineer_features(data)
    data = data.dropna(subset=["TotalCharges"])

    X = data.drop(columns=[target_column, id_column], errors="ignore")
    y = data[target_column].map({"Yes": 1, "No": 0}).astype("uint8")

    numerical_columns = X.select_dtypes(include=["int64", "float64"]).columns
    nominal_columns = X.select_dtypes(include=["object", "bool", "string"]).columns
    ordinal_columns = X.select_dtypes(include=["category"]).columns
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler())
        ]
    )

    nominal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    ordinal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_columns),
            ("nom", nominal_transformer, nominal_columns),
            ("ord", ordinal_transformer, ordinal_columns),
        ]
    )

    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    output_dir = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train_out = (
        X_train_preprocessed.toarray()
        if sparse.issparse(X_train_preprocessed)
        else X_train_preprocessed
    )
    X_test_out = (
        X_test_preprocessed.toarray()
        if sparse.issparse(X_test_preprocessed)
        else X_test_preprocessed
    )
    np.save(output_dir / "X_train_preprocessed.npy", X_train_out)
    np.save(output_dir / "X_test_preprocessed.npy", X_test_out)
    np.save(output_dir / "y_train.npy", y_train.values)
    np.save(output_dir / "y_test.npy", y_test.values)

    raw_dir = Path(os.getenv("DATA_RAW_DIR", "data/raw"))
    submission_test_path = raw_dir / "test.csv"
    if submission_test_path.exists():
        submission_data = _engineer_features(pd.read_csv(submission_test_path))
        submission_data = submission_data.dropna(subset=["TotalCharges"])
        X_submission = submission_data.drop(columns=[target_column], errors="ignore")
        X_submission_features = X_submission.drop(columns=[id_column], errors="ignore")
        X_submission_preprocessed = preprocessor.transform(X_submission_features)
        X_sub_out = (
            X_submission_preprocessed.toarray()
            if sparse.issparse(X_submission_preprocessed)
            else X_submission_preprocessed
        )
        np.save(output_dir / "X_submission_preprocessed.npy", X_sub_out)


if __name__ == "__main__":
    load_dotenv(override=True)
    data = pd.read_csv(Path(os.getenv("DATA_RAW_DIR", "data/raw")) / "train.csv")
    preprocess_data(data)
    print("Data preprocessed successfully")
