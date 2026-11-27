"""Model-loading and inference logic, kept separate from Django views."""

from pathlib import Path

import joblib
import pandas as pd

from ml.feature_schema import FEATURE_COLUMNS
from .models import Patient


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "ml"
    / "artifacts"
    / "heart_disease_pipeline.joblib"
)


class PredictionServiceError(Exception):
    """Raised when the trained pipeline cannot produce a prediction."""


def predict_disease(patient: Patient):
    """Return the predicted class and positive-class model probability."""
    if not MODEL_PATH.is_file():
        raise PredictionServiceError(
            "The prediction model is not available. Train it with "
            "'python -m ml.train_model' and try again."
        )

    try:
        pipeline = joblib.load(MODEL_PATH)
    except (OSError, EOFError, ValueError, ImportError) as exc:
        raise PredictionServiceError(
            "The saved prediction model could not be loaded. Retrain the model "
            "with 'python -m ml.train_model'."
        ) from exc

    features = pd.DataFrame(
        [patient.as_feature_dict()],
        columns=FEATURE_COLUMNS,
    )
    try:
        predicted_class = int(pipeline.predict(features)[0])
        class_probabilities = pipeline.predict_proba(features)[0]
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise PredictionServiceError(
            "The saved model does not support the expected input features. "
            "Retrain the model with 'python -m ml.train_model'."
        ) from exc

    classes = list(pipeline.classes_)
    if 1 not in classes or predicted_class not in (0, 1):
        raise PredictionServiceError(
            "The saved model returned an unsupported prediction class."
        )

    positive_probability = float(class_probabilities[classes.index(1)])
    if not 0 <= positive_probability <= 1:
        raise PredictionServiceError(
            "The saved model returned an invalid probability."
        )
    return bool(predicted_class), positive_probability
