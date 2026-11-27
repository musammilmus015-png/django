"""Page views for authentication, dashboard, predictions, and history."""

import logging

from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.db.models import Count, Q
from django.shortcuts import get_object_or_404, redirect, render

from .forms import PatientForm, RegistrationForm
from .models import Prediction
from .services import PredictionServiceError, predict_disease


logger = logging.getLogger(__name__)


def home(request):
    return render(request, "index.html")


def about(request):
    return render(request, "about.html")


def register(request):
    if request.user.is_authenticated:
        return redirect("dashboard")

    form = RegistrationForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        user = form.save()
        login(request, user)
        return redirect("dashboard")
    return render(request, "register.html", {"form": form})


@login_required
def dashboard(request):
    user_predictions = Prediction.objects.filter(user=request.user)
    stats = user_predictions.aggregate(
        total=Count("pk"),
        positive=Count("pk", filter=Q(predicted_result=True)),
        negative=Count("pk", filter=Q(predicted_result=False)),
    )
    context = {
        "total_predictions": stats["total"],
        "positive_predictions": stats["positive"],
        "negative_predictions": stats["negative"],
        "recent_predictions": user_predictions.select_related("patient")[:8],
        "chart_data": {
            "positive": stats["positive"],
            "negative": stats["negative"],
        },
    }
    return render(request, "dashboard.html", context)


@login_required
def prediction_create(request):
    form = PatientForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        patient = form.save(commit=False)
        try:
            predicted_result, positive_probability = predict_disease(patient)
        except PredictionServiceError as exc:
            logger.exception("Could not generate a heart disease prediction.")
            messages.error(request, str(exc))
            return render(
                request,
                "prediction.html",
                {"form": form},
                status=503,
            )

        with transaction.atomic():
            patient.save()
            prediction = Prediction.objects.create(
                user=request.user,
                patient=patient,
                predicted_result=predicted_result,
                model_probability=positive_probability,
            )
        return redirect("prediction_result", pk=prediction.pk)

    return render(request, "prediction.html", {"form": form})


@login_required
def prediction_result(request, pk):
    prediction = get_object_or_404(
        Prediction.objects.select_related("patient"),
        pk=pk,
        user=request.user,
    )
    return render(request, "result.html", {"prediction": prediction})


@login_required
def prediction_history(request):
    predictions = Prediction.objects.filter(user=request.user).select_related(
        "patient"
    )
    query = request.GET.get("q", "").strip()
    if query:
        normalized_query = query.casefold()
        if normalized_query in {"positive", "negative"}:
            predictions = predictions.filter(
                predicted_result=(normalized_query == "positive")
            )
        elif query.isdecimal():
            predictions = predictions.filter(
                Q(pk=int(query)) | Q(patient__age=int(query))
            )
        else:
            predictions = predictions.none()

    return render(
        request,
        "history.html",
        {"predictions": predictions, "search_query": query},
    )


@login_required
def prediction_delete(request, pk):
    prediction = get_object_or_404(Prediction, pk=pk, user=request.user)
    if request.method == "POST":
        with transaction.atomic():
            prediction.patient.delete()
        messages.success(request, "Prediction record deleted.")
        return redirect("prediction_history")
    return render(request, "prediction_confirm_delete.html", {"prediction": prediction})
