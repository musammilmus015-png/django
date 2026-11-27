"""Local-only SQLite settings for previewing the app without MySQL."""

from .settings import *  # noqa: F403


DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "preview.sqlite3",
    }
}
