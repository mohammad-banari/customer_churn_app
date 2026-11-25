"""Train an independently evaluated baseline churn model from a local labeled CSV.

This script is a new reproducible baseline, not a reproduction of the
original notebook. Supply the original hyperparameters if known.
The deployed model is never overwritten by this script.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

FEATURE_NAMES = [
    "Age", "NumOfProducts", "EstimatedSalary", "Balance", "CreditScore", "Tenure"
]
ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Local labeled CSV")
    parser.add_argument("--target", default="Exited", help="Binary target column")
    parser.add_argument("--params", type=Path, help="Optional JSON of original RF hyperparameters")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1")

    data = pd.read_csv(args.data)
    required = FEATURE_NAMES + [args.target]
    missing = [column for column in required if column not in data.columns]
    if missing:
        parser.error("Missing columns: " + ", ".join(missing))
    if args.target in FEATURE_NAMES:
        parser.error("Target must not be a model input feature")

    X = data[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(data[args.target], errors="coerce")
    if not np.isfinite(X.to_numpy(dtype=float)).all() or not np.isfinite(y.to_numpy(dtype=float)).all():
        parser.error("Missing or invalid values: apply the original cleaning procedure first")
    if set(y.unique()) != {0, 1}:
        parser.error("Target must contain both binary classes 0 and 1")
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # Explicit baseline choices; replace using the original tuning parameters.
    params = {"n_estimators": 100, "random_state": args.seed, "n_jobs": -1}
    if args.params:
        with args.params.open(encoding="utf-8") as file:
            overrides = json.load(file)
        if not isinstance(overrides, dict):
            parser.error("--params must contain a JSON object")
        params.update(overrides)

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    predicted = model.predict(X_test)
    positive_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X_test)[:, positive_index]
    metrics = {
        "scikit_learn_version": sklearn.__version__,
        "feature_names": FEATURE_NAMES,
        "target": args.target,
        "split_seed": args.seed,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "model_params": model.get_params(),
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision": float(precision_score(y_test, predicted, zero_division=0)),
        "recall": float(recall_score(y_test, predicted, zero_division=0)),
        "f1": float(f1_score(y_test, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }

    output = ROOT / "artifacts"
    output.mkdir(exist_ok=True)
    candidate = output / "random_forest_model.candidate.pkl"
    with candidate.open("wb") as file:
        pickle.dump(model, file)
    with (output / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Candidate model saved to {candidate}")
    print("Review held-out metrics and validate the candidate before replacing the deployed model.")


if __name__ == "__main__":
    main()
}
