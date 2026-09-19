"""Convert the trained cancer ANN for lightweight deployment.

Run this on your computer, in the folder that contains ANN.keras and
scaler.pk (both saved at the end of Cancer_Prediction.ipynb):

    python convert_to_tflite.py

It creates:
    ANN.tflite   - the model in TensorFlow Lite format
    scaler.json  - the StandardScaler's means and scales as plain numbers
                   (so the server doesn't need scikit-learn)
and then checks that they give the same predictions as the originals.
"""
import json
import pickle

import numpy as np
import tensorflow as tf

MODEL_PATH = r"C:\Users\Abhijyot Singh Roda\Desktop\Coding Stuff\Python\Projects\Not so random projects\ML Projects\Projects\ANN\Cancer Prediction - ANN\ANN.keras"
SCALER_PATH = r"C:\Users\Abhijyot Singh Roda\Desktop\Coding Stuff\Python\Projects\Not so random projects\ML Projects\Projects\ANN\Cancer Prediction - ANN\scaler.pk"  # saved with pickle.dump in the notebook

model = tf.keras.models.load_model(MODEL_PATH)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

n_features = len(scaler.mean_)
print(f"Scaler expects {n_features} features (the backend sends 25).")
assert n_features == 25, "Feature count is not 25; app.py's FEATURE_ORDER must match."

# 1) Model -> TensorFlow Lite
tflite_model = tf.lite.TFLiteConverter.from_keras_model(model).convert()
with open("ANN.tflite", "wb") as f:
    f.write(tflite_model)
print(f"Saved ANN.tflite ({len(tflite_model) / 1024:.0f} KB)")

# 2) StandardScaler -> plain JSON
with open("scaler.json", "w") as f:
    json.dump({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}, f)
print("Saved scaler.json")

# 3) Sanity check on 20 realistic rows (feature means plus noise)
rng = np.random.default_rng(0)
rows = scaler.mean_ + rng.normal(0, 1.0, size=(20, n_features)) * scaler.scale_

interp = tf.lite.Interpreter(model_content=tflite_model)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]

max_diff = 0.0
label_mismatches = 0
for row in rows:
    row = row.reshape(1, -1)
    keras_prob = float(model.predict(scaler.transform(row), verbose=0)[0][0])

    x = ((row - scaler.mean_) / scaler.scale_).astype(np.float32)
    interp.set_tensor(inp["index"], x)
    interp.invoke()
    tflite_prob = float(interp.get_tensor(out["index"])[0][0])

    max_diff = max(max_diff, abs(keras_prob - tflite_prob))
    label_mismatches += int((keras_prob > 0.5) != (tflite_prob > 0.5))

print(f"Max probability difference over 20 rows: {max_diff:.2e}")
print(f"Rows where the predicted class differs : {label_mismatches}")
print("Looks good." if max_diff < 1e-4 and label_mismatches == 0 else "Check these numbers before deploying.")
