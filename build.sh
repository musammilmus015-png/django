#!/usr/bin/env bash
set -o errexit

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python manage.py collectstatic --no-input
python manage.py migrate --no-input

if [ ! -f "ml/artifacts/heart_disease_pipeline.joblib" ]; then
    python -m ml.train_model
fi
