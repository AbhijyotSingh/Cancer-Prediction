import numpy as np
import joblib
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow the static frontend (served from elsewhere) to call this API

model = tf.keras.models.load_model("ann.keras")
scaler = joblib.load("scaler.pk")

# Must match the exact column order used in train_and_save.py / the notebook.
FEATURE_ORDER = [
    "mean_radius", "mean_texture", "mean_perimeter", "mean_area",
    "mean_smoothness", "mean_compactness", "mean_concavity",
    "mean_concave_points", "mean_symmetry",
    "radius_error", "perimeter_error", "area_error",
    "compactness_error", "concavity_error", "concave_points_error",
    "worst_radius", "worst_texture", "worst_perimeter", "worst_area",
    "worst_smoothness", "worst_compactness", "worst_concavity",
    "worst_concave_points", "worst_symmetry", "worst_fractal_dimension",
]


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Cell Signal backend is awake"})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Expected a JSON body with 25 feature values"}), 400

    missing = [key for key in FEATURE_ORDER if key not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        row = [float(payload[key]) for key in FEATURE_ORDER]
    except (TypeError, ValueError):
        return jsonify({"error": "All fields must be numeric"}), 400

    X = scaler.transform(np.array(row).reshape(1, -1))
    probability = float(model.predict(X, verbose=0)[0][0])
    predicted_outcome = int(probability > 0.5)

    return jsonify({
        "Predicted_outcome": predicted_outcome,
        "probability": probability,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
