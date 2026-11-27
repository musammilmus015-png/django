"""Django admin configuration for prediction records."""

from django.contrib import admin

from .models import Patient, Prediction


@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ("id", "age", "sex", "created_date")
    list_filter = ("sex", "created_date")
    search_fields = ("id", "age")


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "user",
        "predicted_result",
        "model_probability",
        "created_date",
    )
    list_filter = ("predicted_result", "created_date")
    search_fields = ("user__username", "id")
    readonly_fields = ("created_date",)
