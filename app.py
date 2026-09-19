import json
import os
import threading

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore  # small runtime used on Render
except ImportError:  # local development with full TensorFlow installed
    import tensorflow as tf

    Interpreter = tf.lite.Interpreter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app)

# Label direction. In testing, typical malignant cases came out as class 0 and
# typical benign cases as class 1, i.e. the model's output of 1 means BENIGN in
# this dataset, despite the column name benign_0__mal_1. So the output is flipped
# before it is sent to the page. Set to True if check_samples.py shows that rows
# with target = 1 have LARGE measurements (mean radius around 17 or more).
MODEL_OUTPUT_1_IS_MALIGNANT = False

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

interpreter = None
input_index = output_index = None
scaler_mean = scaler_scale = None
model_error = None
lock = threading.Lock()  # a TFLite interpreter is not thread-safe

try:
    with open(os.path.join(BASE_DIR, "scaler.json")) as f:
        scaler_data = json.load(f)
    scaler_mean = np.array(scaler_data["mean"], dtype=np.float64)
    scaler_scale = np.array(scaler_data["scale"], dtype=np.float64)

    interpreter = Interpreter(model_path=os.path.join(BASE_DIR, "ANN.tflite"))
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
except Exception as exc:  # noqa: BLE001
    model_error = f"{type(exc).__name__}: {exc}"


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "Cell Signal backend is awake",
        "model_ready": interpreter is not None,
        "model_error": model_error,
    })


@app.route("/predict", methods=["POST"])
def predict():
    if interpreter is None:
        return jsonify({"error": f"The model failed to load: {model_error}"}), 500

    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Expected a JSON body with 25 feature values"}), 400

    missing = [key for key in FEATURE_ORDER if key not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        row = np.array([float(payload[key]) for key in FEATURE_ORDER], dtype=np.float64)
    except (TypeError, ValueError):
        return jsonify({"error": "All fields must be numeric"}), 400

    # Same as StandardScaler.transform
    X = ((row - scaler_mean) / scaler_scale).astype(np.float32).reshape(1, -1)

    with lock:
        interpreter.set_tensor(input_index, X)
        interpreter.invoke()
        raw_output = float(interpreter.get_tensor(output_index)[0][0])

    # The frontend expects: Predicted_outcome 1 = malignant, and
    # probability = chance of malignant.
    probability = raw_output if MODEL_OUTPUT_1_IS_MALIGNANT else 1.0 - raw_output

    return jsonify({
        "Predicted_outcome": int(probability > 0.5),
        "probability": probability,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
