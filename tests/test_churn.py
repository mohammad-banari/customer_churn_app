"""Smoke tests for the saved bank customer churn classifier."""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODEL_PATH = Path(os.environ.get(
    "CHURN_MODEL_PATH",
    str(Path(__file__).resolve().parents[1] / "random_forest_model.pkl"),
))
FEATURE_NAMES = [
    "Age", "NumOfProducts", "EstimatedSalary", "Balance", "CreditScore", "Tenure"
]


@pytest.fixture(scope="module")
def model():
    # Only load a trusted repository artifact or locally trained candidate.
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


@pytest.fixture
def sample_customers():
    return pd.DataFrame(
        [
            [40, 1, 50000, 60000, 650, 5],
            [52, 2, 82000, 125000, 580, 8],
            [31, 1, 42000, 0, 720, 2],
        ],
        columns=FEATURE_NAMES,
    )


def test_model_loading(model):
    assert model is not None
    assert hasattr(model, "predict")


def test_single_prediction(model, sample_customers):
    prediction = model.predict(sample_customers.iloc[[0]])
    assert len(prediction) == 1
    assert prediction[0] in [0, 1]


def test_batch_prediction(model, sample_customers):
    predictions = model.predict(sample_customers)
    assert len(predictions) == len(sample_customers)
    assert np.isin(predictions, [0, 1]).all()


def test_prediction_probability(model, sample_customers):
    probabilities = model.predict_proba(sample_customers)
    assert probabilities.shape == (len(sample_customers), 2)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    assert np.allclose(probabilities.sum(axis=1), 1)
