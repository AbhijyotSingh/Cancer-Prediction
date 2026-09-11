# Cell Signal

A small deployable version of the ANN from `Cancer_Prediction.ipynb`: a form
for the 25 tumor-nuclei measurements the model was trained on, plus a Flask
API that loads the trained Keras model and its scaler to return a
benign/malignant estimate.

## Files
- `index.html`, `css.css`, `app.js` — the frontend
- `app.py` — Flask backend, exposes `POST /predict`
- `train_and_save.py` — trains the same architecture as the notebook and
  saves `cancer_model.keras` + `scaler.pkl` (run this first; those two files
  aren't included here since only the notebook's code was available, not its
  trained weights)
- `requirements.txt` — backend dependencies

## Run it locally

```bash
pip install -r requirements.txt
python train_and_save.py    # trains + saves cancer_model.keras and scaler.pkl
python app.py                # starts the API on http://127.0.0.1:5000
```

Then open `index.html` in a browser (or serve the folder with any static
file server). Click "Use sample case" to fill in a known example, then "Run
analysis".

## Deploying for real
- Backend: push this folder to Render, Railway, Fly.io, or similar, same as
  the phishing-scanner example you shared. Make sure `cancer_model.keras`
  and `scaler.pkl` are included in the deploy (either commit them, or run
  `train_and_save.py` as a build step).
- Frontend: host `index.html`/`css.css`/`app.js` as a static site (GitHub
  Pages, Netlify, Vercel, or the same server as the backend), and update
  `BACKEND_URL` in `app.js` to your deployed backend's URL.
- CORS is already enabled on the backend (`flask-cors`) so the two can live
  on different domains.

## Important caveat
`train_and_save.py` trains on `sklearn.datasets.load_breast_cancer()`, which
is the same public 569-row Wisconsin dataset your CSV is built from — this
lets you reproduce a working model without your original local file path.
If you have your own `cancer_classification.csv` and want to train on that
exact file instead, swap the "Load data" block in `train_and_save.py` for
your `pd.read_csv(...)` call; everything downstream (feature order, model
architecture) stays the same.

This tool is a demonstration model trained on a small public dataset. It is
not a medical device and shouldn't be presented as a diagnostic tool.
