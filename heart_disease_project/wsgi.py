"""WSGI config for the heart disease prediction project."""

import os

from django.core.wsgi import get_wsgi_application


os.environ.setdefault("DJANGO_SETTINGS_MODULE", "heart_disease_project.settings")

application = get_wsgi_application()
