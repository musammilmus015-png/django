"""Shared feature names and categories for training and web-form integration."""

FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

FEATURE_CHOICES = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {
        1: "Typical angina",
        2: "Atypical angina",
        3: "Non-anginal pain",
        4: "Asymptomatic",
    },
    "fbs": {0: "No", 1: "Yes"},
    "restecg": {
        0: "Normal",
        1: "ST-T wave abnormality",
        2: "Left ventricular hypertrophy",
    },
    "exang": {0: "No", 1: "Yes"},
    "slope": {
        1: "Upsloping",
        2: "Flat",
        3: "Downsloping",
    },
    "ca": {0: "0", 1: "1", 2: "2", 3: "3"},
    "thal": {
        3: "Normal",
        6: "Fixed defect",
        7: "Reversible defect",
    },
}
