"""Database models for patient inputs and prediction history."""

from decimal import Decimal

from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models

from ml.feature_schema import FEATURE_CHOICES


def _choices_for(feature):
    return [(value, label) for value, label in FEATURE_CHOICES[feature].items()]


class Patient(models.Model):
    age = models.PositiveSmallIntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(120)]
    )
    sex = models.PositiveSmallIntegerField(choices=_choices_for("sex"))
    chest_pain_type = models.PositiveSmallIntegerField(choices=_choices_for("cp"))
    blood_pressure = models.PositiveSmallIntegerField(
        validators=[MinValueValidator(50), MaxValueValidator(300)]
    )
    cholesterol = models.PositiveSmallIntegerField(
        validators=[MinValueValidator(50), MaxValueValidator(700)]
    )
    fasting_blood_sugar = models.PositiveSmallIntegerField(
        choices=_choices_for("fbs")
    )
    resting_ecg = models.PositiveSmallIntegerField(choices=_choices_for("restecg"))
    maximum_heart_rate = models.PositiveSmallIntegerField(
        validators=[MinValueValidator(30), MaxValueValidator(250)]
    )
    exercise_angina = models.PositiveSmallIntegerField(choices=_choices_for("exang"))
    oldpeak = models.DecimalField(
        max_digits=4,
        decimal_places=1,
        validators=[MinValueValidator(-5), MaxValueValidator(20)],
    )
    slope = models.PositiveSmallIntegerField(choices=_choices_for("slope"))
    number_of_vessels = models.PositiveSmallIntegerField(
        choices=_choices_for("ca")
    )
    thalassemia = models.PositiveSmallIntegerField(choices=_choices_for("thal"))
    created_date = models.DateTimeField(auto_now_add=True)

    def as_feature_dict(self):
        """Return patient values under the feature names used during training."""
        return {
            "age": self.age,
            "sex": self.sex,
            "cp": self.chest_pain_type,
            "trestbps": self.blood_pressure,
            "chol": self.cholesterol,
            "fbs": self.fasting_blood_sugar,
            "restecg": self.resting_ecg,
            "thalach": self.maximum_heart_rate,
            "exang": self.exercise_angina,
            "oldpeak": float(self.oldpeak),
            "slope": self.slope,
            "ca": self.number_of_vessels,
            "thal": self.thalassemia,
        }

    def __str__(self):
        return f"Patient input {self.pk} (age {self.age})"


class Prediction(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="heart_predictions",
    )
    patient = models.OneToOneField(
        Patient,
        on_delete=models.CASCADE,
        related_name="prediction",
    )
    predicted_result = models.BooleanField()
    model_probability = models.DecimalField(
        max_digits=7,
        decimal_places=6,
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        help_text="Model-estimated probability for the positive class; not a clinical risk score.",
    )
    created_date = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_date"]

    @property
    def model_probability_percent(self):
        return self.model_probability * Decimal("100")

    def __str__(self):
        result = "Positive" if self.predicted_result else "Negative"
        return f"{result} prediction #{self.pk}"
