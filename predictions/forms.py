"""Validated forms for registration and patient input."""

from django import forms
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.models import User

from .models import Patient


class RegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta(UserCreationForm.Meta):
        model = User
        fields = ("username", "email")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["email"].widget.attrs["class"] = "form-control"
        for field in self.fields.values():
            field.widget.attrs.setdefault("class", "form-control")


class LoginForm(AuthenticationForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs["class"] = "form-control"


class PatientForm(forms.ModelForm):
    class Meta:
        model = Patient
        fields = [
            "age",
            "sex",
            "chest_pain_type",
            "blood_pressure",
            "cholesterol",
            "fasting_blood_sugar",
            "resting_ecg",
            "maximum_heart_rate",
            "exercise_angina",
            "oldpeak",
            "slope",
            "number_of_vessels",
            "thalassemia",
        ]
        labels = {
            "age": "Age",
            "sex": "Sex",
            "chest_pain_type": "Chest pain type",
            "blood_pressure": "Resting blood pressure (mm Hg)",
            "cholesterol": "Serum cholesterol (mg/dL)",
            "fasting_blood_sugar": "Fasting blood sugar > 120 mg/dL",
            "resting_ecg": "Resting ECG",
            "maximum_heart_rate": "Maximum heart rate",
            "exercise_angina": "Exercise-induced angina",
            "oldpeak": "ST depression (Oldpeak)",
            "slope": "Slope of peak exercise ST segment",
            "number_of_vessels": "Number of major vessels (0-3)",
            "thalassemia": "Thalassemia",
        }
        widgets = {
            "age": forms.NumberInput(attrs={"min": 1, "max": 120, "step": 1}),
            "blood_pressure": forms.NumberInput(
                attrs={"min": 50, "max": 300, "step": 1}
            ),
            "cholesterol": forms.NumberInput(
                attrs={"min": 50, "max": 700, "step": 1}
            ),
            "maximum_heart_rate": forms.NumberInput(
                attrs={"min": 30, "max": 250, "step": 1}
            ),
            "oldpeak": forms.NumberInput(
                attrs={"min": -5, "max": 20, "step": "0.1"}
            ),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs.setdefault("class", "form-select")
        for name in (
            "age",
            "blood_pressure",
            "cholesterol",
            "maximum_heart_rate",
            "oldpeak",
        ):
            self.fields[name].widget.attrs["class"] = "form-control"
