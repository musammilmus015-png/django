"""Download the UCI Cleveland data, compare classifiers, and save the best pipeline."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import joblib
import numpy as np

from ml.feature_schema import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "ml" / "data"
ARTIFACT_DIR = PROJECT_ROOT / "ml" / "artifacts"
RANDOM_STATE = 42


def load_uci_cleveland_data():
    """Fetch UCI dataset 45 and retain its 13 inputs and disease target."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    source_path = DATA_DIR / "uci_heart_disease_source.csv"
    source_url = "https://archive.ics.uci.edu/static/public/45/data.csv"
    if source_path.exists():
        raw_data = pd.read_csv(source_path, na_values="?")
    else:
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=45)
        raw_data = dataset.data.features.copy()
        raw_data["num"] = dataset.data.targets["num"]
        raw_data.to_csv(source_path, index=False)

    missing_features = set(FEATURE_COLUMNS) - set(raw_data.columns)
    if missing_features:
        raise ValueError(f"UCI dataset is missing expected features: {sorted(missing_features)}")
    if "num" not in raw_data.columns:
        raise ValueError("UCI dataset is missing the expected 'num' target.")

    features = raw_data[FEATURE_COLUMNS].replace("?", np.nan).copy()
    for column in FEATURE_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="coerce")

    target = pd.to_numeric(raw_data["num"].replace("?", np.nan), errors="coerce")
    frame = features.copy()
    frame["target"] = target

    rows_before_deduplication = len(frame)
    frame = frame.drop_duplicates().copy()
    duplicates_removed = rows_before_deduplication - len(frame)

    rows_before_target_cleanup = len(frame)
    frame = frame.dropna(subset=["target"]).copy()
    missing_targets_removed = rows_before_target_cleanup - len(frame)
    frame["target"] = frame["target"].gt(0).astype("int8")

    if frame.empty:
        raise ValueError("UCI dataset has no rows remaining after target cleanup.")
    if frame["target"].nunique() != 2:
        raise ValueError("Expected both disease-negative and disease-positive target classes.")

    frame.to_csv(DATA_DIR / "heart_disease_cleveland.csv", index=False)
    profile = {
        "source": source_url,
        "rows_after_cleaning": int(len(frame)),
        "duplicates_removed": int(duplicates_removed),
        "rows_with_missing_target_removed": int(missing_targets_removed),
        "feature_columns": FEATURE_COLUMNS,
        "target_definition": "num == 0 -> 0; num > 0 -> 1",
        "missing_values_by_feature": {
            column: int(count)
            for column, count in frame[FEATURE_COLUMNS].isna().sum().items()
        },
        "class_counts": {
            str(label): int(count)
            for label, count in frame["target"].value_counts().sort_index().items()
        },
        "numeric_summary": json.loads(
            frame[NUMERIC_FEATURES].describe().to_json()
        ),
    }
    return frame, profile


def make_preprocessor():
    """Build imputation, scaling, and one-hot encoding as part of model inference."""
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def make_models():
    """Return each candidate classifier with the identical preprocessing pipeline."""
    classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Support Vector Machine": CalibratedClassifierCV(
            estimator=SVC(
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            method="sigmoid",
            cv=3,
        ),
    }
    return {
        name: Pipeline(
            [
                ("preprocessor", make_preprocessor()),
                ("classifier", classifier),
            ]
        )
        for name, classifier in classifiers.items()
    }


def train_and_evaluate():
    frame, profile = load_uci_cleveland_data()
    x = frame[FEATURE_COLUMNS]
    y = frame["target"]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    models = make_models()
    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    results = {}
    for name, model in models.items():
        cv_result = cross_validate(
            model,
            x_train,
            y_train,
            cv=cross_validation,
            scoring=scoring,
            error_score="raise",
        )
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        results[name] = {
            "cross_validation": {
                metric: {
                    "mean": float(np.mean(cv_result[f"test_{metric}"])),
                    "std": float(np.std(cv_result[f"test_{metric}"])),
                }
                for metric in scoring
            },
            "test": {
                "accuracy": float(accuracy_score(y_test, predictions)),
                "precision": float(
                    precision_score(y_test, predictions, zero_division=0)
                ),
                "recall": float(recall_score(y_test, predictions, zero_division=0)),
                "f1": float(f1_score(y_test, predictions, zero_division=0)),
                "confusion_matrix": confusion_matrix(
                    y_test, predictions, labels=[0, 1]
                ).tolist(),
            },
        }

    selected_name = max(
        results,
        key=lambda name: (
            results[name]["cross_validation"]["f1"]["mean"],
            results[name]["cross_validation"]["recall"]["mean"],
        ),
    )
    selected_model = models[selected_name]
    selected_predictions = selected_model.predict(x_test)
    selected_confusion = confusion_matrix(
        y_test, selected_predictions, labels=[0, 1]
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_model, ARTIFACT_DIR / "heart_disease_pipeline.joblib")

    report = {
        "selection_metric": "highest mean 5-fold cross-validation F1 on training data; recall breaks ties",
        "selected_model": selected_name,
        "random_state": RANDOM_STATE,
        "test_size": 0.2,
        "profile": profile,
        "models": results,
    }
    (ARTIFACT_DIR / "evaluation_metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        selected_confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No disease", "Disease"],
        yticklabels=["No disease", "Disease"],
    )
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.title(f"{selected_name} — held-out test set")
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    print(f"Selected model (training CV F1): {selected_name}")
    print(f"Cleaned Cleveland records: {len(frame)}")
    print("Held-out test metrics:")
    for name, result in results.items():
        print(f"  {name}: {result['test']}")
    print("Saved pipeline and reports under ml/artifacts.")


if __name__ == "__main__":
    train_and_evaluate()
