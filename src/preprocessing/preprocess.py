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


def preprocess_data(data: pd.DataFrame, target_column: str | None = None):
    target_column = target_column or os.getenv("TARGET_COLUMN")
    id_column = os.getenv("ID_COLUMN", "id")
    # FIXME add data transformation here if needed
    X = data.drop(columns=[target_column, id_column], errors="ignore")
    # FIXME change it as needed
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
        submission_data = pd.read_csv(submission_test_path)
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
