"""URL patterns for the prediction application."""

from django.contrib.auth import views as auth_views
from django.urls import path

from .forms import LoginForm
from . import views


urlpatterns = [
    path("", views.home, name="home"),
    path("about/", views.about, name="about"),
    path("register/", views.register, name="register"),
    path(
        "login/",
        auth_views.LoginView.as_view(
            template_name="login.html",
            authentication_form=LoginForm,
        ),
        name="login",
    ),
    path(
        "logout/",
        auth_views.LogoutView.as_view(),
        name="logout",
    ),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("prediction/", views.prediction_create, name="prediction_create"),
    path("history/", views.prediction_history, name="prediction_history"),
    path(
        "history/<int:pk>/",
        views.prediction_result,
        name="prediction_result",
    ),
    path(
        "history/<int:pk>/delete/",
        views.prediction_delete,
        name="prediction_delete",
    ),
]
