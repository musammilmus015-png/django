"""ASGI config for the heart disease prediction project."""

import os

from django.core.asgi import get_asgi_application


os.environ.setdefault("DJANGO_SETTINGS_MODULE", "heart_disease_project.settings")

application = get_asgi_application()
