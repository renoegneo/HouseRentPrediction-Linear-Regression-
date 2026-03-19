from flask import Flask, render_template, request, send_from_directory
import os
import pandas as pd

from test_model import predict_from_input, get_r2_score

DATAFILE = "House_Rent_Dataset.csv"
PLOT_FILE = "regression_plot.png"

app = Flask(__name__)


def load_dataset():
    return pd.read_csv(DATAFILE)


DATASET = load_dataset()
DATASET_COLUMNS = DATASET.columns.tolist()
DATASET_HEAD = DATASET.head(10).to_dict(orient="records")
UNUSED_COLUMNS = [col for col in ["Posted On", "Area Locality", "Floor", "Rent"] if col in DATASET_COLUMNS]


def build_model_ready_preview(df):
    target_columns = ["Size", "BHK", "Bathroom", "Floor", "City", "Furnishing Status"]
    existing = [col for col in target_columns if col in df.columns]
    preview = df[existing].dropna()
    return preview


MODEL_READY = build_model_ready_preview(DATASET)
MODEL_READY_COLUMNS = MODEL_READY.columns.tolist()
MODEL_READY_HEAD = MODEL_READY.head(5).to_dict(orient="records")


def parse_int(value, field_name, errors):
    try:
        return int(value)
    except (TypeError, ValueError):
        errors.append(f"{field_name} must be an integer.")
        return None


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        dataset_columns=DATASET_COLUMNS,
        dataset_head=DATASET_HEAD,
        model_ready_columns=MODEL_READY_COLUMNS,
        model_ready_head=MODEL_READY_HEAD,
        unused_columns=UNUSED_COLUMNS,
        r2_score=round(get_r2_score(), 4),
        prediction=None,
        errors=[],
        form_data={},
    )


@app.route("/predict", methods=["POST"])
def predict():
    form_data = {
        "size": request.form.get("size", "").strip(),
        "bhk": request.form.get("bhk", "").strip(),
        "bathroom": request.form.get("bathroom", "").strip(),
        "floor": request.form.get("floor", "").strip(),
        "city": request.form.get("city", "").strip(),
        "furnishing_status": request.form.get("furnishing_status", "").strip(),
    }

    errors = []
    size = parse_int(form_data["size"], "Size", errors)
    bhk = parse_int(form_data["bhk"], "BHK", errors)
    bathroom = parse_int(form_data["bathroom"], "Bathroom", errors)

    for key, label in [
        ("floor", "Floor"),
        ("city", "City"),
        ("furnishing_status", "Furnishing Status"),
    ]:
        if not form_data[key]:
            errors.append(f"{label} is required.")

    prediction = None
    if not errors:
        payload = {
            "Size": size,
            "BHK": bhk,
            "Bathroom": bathroom,
            "Floor": form_data["floor"],
            "City": form_data["city"],
            "Furnishing Status": form_data["furnishing_status"],
        }
        prediction = round(predict_from_input(payload), 2)

    return render_template(
        "index.html",
        dataset_columns=DATASET_COLUMNS,
        dataset_head=DATASET_HEAD,
        model_ready_columns=MODEL_READY_COLUMNS,
        model_ready_head=MODEL_READY_HEAD,
        unused_columns=UNUSED_COLUMNS,
        r2_score=round(get_r2_score(), 4),
        prediction=prediction,
        errors=errors,
        form_data=form_data,
    )


@app.route("/regression_plot.png")
def regression_plot():
    return send_from_directory(os.getcwd(), PLOT_FILE)


if __name__ == "__main__":
    app.run(debug=True)
