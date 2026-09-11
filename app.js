const BACKEND_URL = "http://127.0.0.1:5000";

const FEATURES = [
  { key: "mean_radius", label: "Radius", group: "mean", sample: 17.99 },
  { key: "mean_texture", label: "Texture", group: "mean", sample: 10.38 },
  { key: "mean_perimeter", label: "Perimeter", group: "mean", sample: 122.8 },
  { key: "mean_area", label: "Area", group: "mean", sample: 1001.0 },
  { key: "mean_smoothness", label: "Smoothness", group: "mean", sample: 0.1184 },
  { key: "mean_compactness", label: "Compactness", group: "mean", sample: 0.2776 },
  { key: "mean_concavity", label: "Concavity", group: "mean", sample: 0.3001 },
  { key: "mean_concave_points", label: "Concave points", group: "mean", sample: 0.1471 },
  { key: "mean_symmetry", label: "Symmetry", group: "mean", sample: 0.2419 },

  { key: "radius_error", label: "Radius", group: "error", sample: 1.095 },
  { key: "perimeter_error", label: "Perimeter", group: "error", sample: 8.589 },
  { key: "area_error", label: "Area", group: "error", sample: 153.4 },
  { key: "compactness_error", label: "Compactness", group: "error", sample: 0.04904 },
  { key: "concavity_error", label: "Concavity", group: "error", sample: 0.05373 },
  { key: "concave_points_error", label: "Concave points", group: "error", sample: 0.01587 },

  { key: "worst_radius", label: "Radius", group: "worst", sample: 25.38 },
  { key: "worst_texture", label: "Texture", group: "worst", sample: 17.33 },
  { key: "worst_perimeter", label: "Perimeter", group: "worst", sample: 184.6 },
  { key: "worst_area", label: "Area", group: "worst", sample: 2019.0 },
  { key: "worst_smoothness", label: "Smoothness", group: "worst", sample: 0.1622 },
  { key: "worst_compactness", label: "Compactness", group: "worst", sample: 0.6656 },
  { key: "worst_concavity", label: "Concavity", group: "worst", sample: 0.7119 },
  { key: "worst_concave_points", label: "Concave points", group: "worst", sample: 0.2654 },
  { key: "worst_symmetry", label: "Symmetry", group: "worst", sample: 0.4601 },
  { key: "worst_fractal_dimension", label: "Fractal dimension", group: "worst", sample: 0.1189 },
];

const groupEls = {
  mean: document.getElementById("group-mean"),
  error: document.getElementById("group-error"),
  worst: document.getElementById("group-worst"),
};

FEATURES.forEach((f) => {
  const wrap = document.createElement("div");
  wrap.className = "field";
  wrap.innerHTML = `
    <label for="${f.key}">${f.label}</label>
    <input type="number" step="any" id="${f.key}" name="${f.key}" required>
  `;
  groupEls[f.group].appendChild(wrap);
});

document.getElementById("fillSample").addEventListener("click", () => {
  FEATURES.forEach((f) => {
    document.getElementById(f.key).value = f.sample;
  });
});

const form = document.getElementById("scanForm");
const ticket = document.getElementById("ticket");
const ticketStatus = document.getElementById("ticketStatus");
const ticketScore = document.getElementById("ticketScore");
const ticketNote = document.getElementById("ticketNote");
const runBtn = document.getElementById("runBtn");

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const payload = {};
  FEATURES.forEach((f) => {
    payload[f.key] = parseFloat(document.getElementById(f.key).value);
  });

  runBtn.disabled = true;
  runBtn.textContent = "Scanning…";
  ticket.className = "ticket pending";
  ticketStatus.textContent = "Analyzing sample";
  ticketScore.textContent = "—";
  ticketNote.textContent = "This may take a moment if the server was idle.";

  try {
    const response = await fetch(BACKEND_URL + "/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) throw new Error("Bad response: " + response.status);
    const data = await response.json();

    const isMalignant = data.Predicted_outcome === 1 || data.Predicted_outcome === 1.0;
    const probability = data.probability;

    ticket.className = "ticket " + (isMalignant ? "malignant" : "benign");
    ticketStatus.textContent = isMalignant ? "Reads as malignant" : "Reads as benign";
    ticketScore.textContent =
      probability !== undefined ? `p(malignant) = ${probability.toFixed(4)}` : "";
    ticketNote.textContent = isMalignant
      ? "The measurements pattern-match to malignant cases in the training data. Confirm with a pathologist."
      : "The measurements pattern-match to benign cases in the training data. This is not a clearance.";
  } catch (err) {
    console.error(err);
    ticket.className = "ticket malignant";
    ticketStatus.textContent = "Couldn't reach the model";
    ticketScore.textContent = "";
    ticketNote.textContent = "Check that the backend server is running, then try again.";
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = "Run analysis";
  }
});
