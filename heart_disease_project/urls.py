"""URL configuration for the heart disease prediction project."""

from django.contrib import admin
from django.urls import include, path


urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("predictions.urls")),
]
