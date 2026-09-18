from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models.pkl"
SCALER_PATH = BASE_DIR / "standard_scaler.pkl"


class ArtifactConfigurationError(RuntimeError):
	"""Raised when the supplied artifacts cannot form the trained input."""


def load_pickle(path: Path) -> Any:
	if not path.exists():
		raise ArtifactConfigurationError(f"Missing required artifact: {path.name}")
	with path.open("rb") as file:
		return pickle.load(file)


def load_artifacts() -> tuple[Any, Any, list[str]]:
	model = load_pickle(MODEL_PATH)
	scaler = load_pickle(SCALER_PATH)

	feature_names = getattr(scaler, "feature_names_in_", None)
	if feature_names is None:
		raise ArtifactConfigurationError(
			"standard_scaler.pkl does not contain feature_names_in_; "
			"the trained feature order cannot be verified."
		)

	feature_names = [str(name) for name in feature_names]
	if not feature_names:
		raise ArtifactConfigurationError("The fitted scaler has no input features.")

	return model, scaler, feature_names


try:
	MODEL, SCALER, FEATURE_NAMES = load_artifacts()
	ARTIFACT_ERROR = None
except (OSError, pickle.PickleError, AttributeError, ValueError, TypeError) as error:
	MODEL = SCALER = None
	FEATURE_NAMES = []
	ARTIFACT_ERROR = f"Could not load the prediction artifacts: {error}"
except ArtifactConfigurationError as error:
	MODEL = SCALER = None
	FEATURE_NAMES = []
	ARTIFACT_ERROR = str(error)


def predict(values: dict[str, str]) -> float:
	if ARTIFACT_ERROR:
		raise ArtifactConfigurationError(ARTIFACT_ERROR)
	model = MODEL
	scaler = SCALER
	if model is None or scaler is None:
		raise ArtifactConfigurationError("Prediction artifacts are not loaded.")

	try:
		row = {name: float(values[name]) for name in FEATURE_NAMES}
	except (KeyError, TypeError, ValueError) as error:
		raise ValueError("Enter a valid number for every field.") from error

	frame = pd.DataFrame([row], columns=FEATURE_NAMES)
	transformed = scaler.transform(frame)
	expected_width = getattr(model, "n_features_in_", None)
	if expected_width != transformed.shape[1]:
		raise ArtifactConfigurationError(
			"The fitted scaler produces "
			f"{transformed.shape[1]} features, but the model expects "
			f"{expected_width}. A trained encoder or feature-transformer "
			"pickle is missing; prediction is disabled to avoid guessing "
			"the feature mapping."
		)

	prediction = model.predict(transformed)
	return float(prediction[0])


app = Flask(__name__)
app.config["SECRET_KEY"] = "local-development-key"


@app.get("/")
def index():
	return render_template(
		"index.html",
		feature_names=FEATURE_NAMES,
		artifact_error=ARTIFACT_ERROR,
	)


@app.post("/predict")
def make_prediction():
	try:
		result = predict(request.form)
	except (ArtifactConfigurationError, ValueError) as error:
		flash(str(error), "error")
		return redirect(url_for("index"))

	return render_template("result.html", prediction=result)


if __name__ == "__main__":
	app.run(debug=True)
