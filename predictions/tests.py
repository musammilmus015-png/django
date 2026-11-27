"""Focused tests for patient validation and model inference integration."""

from pathlib import Path
from unittest.mock import Mock, patch

from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase
from django.urls import reverse

from .forms import PatientForm
from .models import Patient, Prediction
from .services import PredictionServiceError, predict_disease


VALID_PATIENT_DATA = {
    "age": 63,
    "sex": 1,
    "chest_pain_type": 1,
    "blood_pressure": 145,
    "cholesterol": 233,
    "fasting_blood_sugar": 1,
    "resting_ecg": 2,
    "maximum_heart_rate": 150,
    "exercise_angina": 0,
    "oldpeak": "2.3",
    "slope": 3,
    "number_of_vessels": 0,
    "thalassemia": 6,
}


class PatientFormTests(SimpleTestCase):
    def test_accepts_valid_patient_inputs(self):
        form = PatientForm(data=VALID_PATIENT_DATA)

        self.assertTrue(form.is_valid(), form.errors)

    def test_rejects_out_of_range_and_unknown_category_values(self):
        invalid_data = {
            **VALID_PATIENT_DATA,
            "age": 121,
            "sex": 2,
        }
        form = PatientForm(data=invalid_data)

        self.assertFalse(form.is_valid())
        self.assertIn("age", form.errors)
        self.assertIn("sex", form.errors)


class PredictionServiceTests(SimpleTestCase):
    def test_passes_training_feature_schema_and_returns_positive_probability(self):
        patient = Patient(**VALID_PATIENT_DATA)
        pipeline = Mock()
        pipeline.predict.return_value = [1]
        pipeline.predict_proba.return_value = [[0.2, 0.8]]
        pipeline.classes_ = [0, 1]

        with (
            patch("predictions.services.MODEL_PATH", Path(__file__)),
            patch("predictions.services.joblib.load", return_value=pipeline),
        ):
            result, probability = predict_disease(patient)

        self.assertIs(result, True)
        self.assertEqual(probability, 0.8)
        input_frame = pipeline.predict.call_args.args[0]
        self.assertEqual(
            list(input_frame.columns),
            [
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
            ],
        )

    def test_reports_missing_model_artifact(self):
        patient = Patient(**VALID_PATIENT_DATA)

        with patch(
            "predictions.services.MODEL_PATH",
            Path(__file__).parent / "missing-model.joblib",
        ):
            with self.assertRaisesMessage(
                PredictionServiceError, "prediction model is not available"
            ):
                predict_disease(patient)


class PublicTemplateTests(TestCase):
    def test_public_pages_render(self):
        for route_name in ("home", "about", "register", "login"):
            with self.subTest(page=route_name):
                response = self.client.get(reverse(route_name))
                self.assertEqual(response.status_code, 200)

    def test_registration_creates_and_authenticates_user(self):
        response = self.client.post(
            reverse("register"),
            data={
                "username": "new-researcher",
                "email": "researcher@example.com",
                "password1": "Uniq-Research-Pass-9843!",
                "password2": "Uniq-Research-Pass-9843!",
            },
        )

        self.assertRedirects(response, reverse("dashboard"))
        self.assertTrue(
            User.objects.filter(username="new-researcher").exists()
        )


class AuthenticatedPageTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="researcher",
            password="test-password-123",
        )
        self.client.force_login(self.user)

    def test_dashboard_prediction_and_history_pages_render(self):
        for route_name in ("dashboard", "prediction_create", "prediction_history"):
            with self.subTest(page=route_name):
                response = self.client.get(reverse(route_name))
                self.assertEqual(response.status_code, 200)

    def test_login_and_logout(self):
        self.client.logout()

        login_response = self.client.post(
            reverse("login"),
            data={"username": "researcher", "password": "test-password-123"},
        )
        self.assertRedirects(login_response, reverse("dashboard"))

        logout_response = self.client.post(reverse("logout"))
        self.assertRedirects(logout_response, reverse("home"))

    @patch("predictions.views.predict_disease", return_value=(True, 0.73))
    def test_valid_submission_saves_patient_and_prediction(self, predict_mock):
        response = self.client.post(
            reverse("prediction_create"),
            data=VALID_PATIENT_DATA,
        )

        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.url.startswith("/history/"))
        prediction = self.user.heart_predictions.get()
        self.assertTrue(prediction.predicted_result)
        self.assertAlmostEqual(float(prediction.model_probability), 0.73)
        self.assertEqual(prediction.patient.age, 63)
        predict_mock.assert_called_once()

        dashboard_response = self.client.get(reverse("dashboard"))
        self.assertEqual(dashboard_response.context["total_predictions"], 1)
        self.assertEqual(dashboard_response.context["positive_predictions"], 1)
        self.assertEqual(dashboard_response.context["negative_predictions"], 0)

        result_response = self.client.get(
            reverse("prediction_result", args=[prediction.pk])
        )
        self.assertEqual(result_response.status_code, 200)
        self.assertContains(result_response, "Positive model output")
        self.assertContains(result_response, "73.0%")
        self.assertContains(result_response, "not a clinically validated")

    def test_user_cannot_view_another_users_prediction(self):
        other_user = User.objects.create_user(
            username="another",
            password="test-password-123",
        )
        patient = Patient.objects.create(**VALID_PATIENT_DATA)
        prediction = Prediction.objects.create(
            user=other_user,
            patient=patient,
            predicted_result=False,
            model_probability=0.2,
        )

        response = self.client.get(
            reverse("prediction_result", args=[prediction.pk])
        )

        self.assertEqual(response.status_code, 404)

    def test_prediction_can_only_be_deleted_by_post(self):
        patient = Patient.objects.create(**VALID_PATIENT_DATA)
        prediction = Prediction.objects.create(
            user=self.user,
            patient=patient,
            predicted_result=False,
            model_probability=0.2,
        )

        confirm_response = self.client.get(
            reverse("prediction_delete", args=[prediction.pk])
        )
        self.assertEqual(confirm_response.status_code, 200)

        delete_response = self.client.post(
            reverse("prediction_delete", args=[prediction.pk])
        )
        self.assertRedirects(delete_response, reverse("prediction_history"))
        self.assertFalse(Prediction.objects.filter(pk=prediction.pk).exists())
        self.assertFalse(Patient.objects.filter(pk=patient.pk).exists())
