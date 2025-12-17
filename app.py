# app.py
import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json

# ---------------------------------------------------
# PATHS
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "waste_4class_baseline.keras")
MAPPING_PATH = os.path.join(BASE_DIR, "models", "class_mapping.json")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------------------------------
# LOAD MODEL + CLASS LABELS
# ---------------------------------------------------
print("📌 Loading Model...")
model = load_model(MODEL_PATH)

with open(MAPPING_PATH, "r") as f:
    class_labels = json.load(f)["class_names"]

print("📌 Loaded classes:", class_labels)

# ---------------------------------------------------
# FLASK
# ---------------------------------------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

IMG_SIZE = (224, 224)   # same as training


# ---------------------------------------------------
# IMAGE PREDICT FUNCTION
# ---------------------------------------------------
def predict_image(img_path):

    img = load_img(img_path, target_size=IMG_SIZE)
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)[0]             # shape: (4,)
    top_idx = np.argmax(preds)
    top_class = class_labels[top_idx]
    top_conf = float(preds[top_idx])

    # return dict of all probabilities
    scores = {label: float(preds[i]) for i, label in enumerate(class_labels)}

    return top_class, top_conf, scores


# ---------------------------------------------------
# ROUTES
# ---------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]
    if file.filename == "":
        return "No file selected"

    # SAVE UPLOADED IMAGE
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # RUN PREDICTION
    pred_class, conf, scores = predict_image(filepath)

    # relative path for HTML <img>
    rel_path = os.path.join("static", "uploads", file.filename)

    return render_template(
        "index.html",
        prediction=pred_class,
        top_conf=conf,
        scores=scores,
        image_path=rel_path
    )


# ---------------------------------------------------
# FEEDBACK ROUTE (optional but working)
# ---------------------------------------------------
@app.route("/feedback", methods=["POST"])
def feedback():
    is_correct = request.form["correct"]
    pred = request.form["prediction"]
    image = request.form["image"]

    log_line = f"{image}, predicted={pred}, correct={is_correct}\n"

    with open(os.path.join("feedback", "feedback_log.csv"), "a") as f:
        f.write(log_line)

    msg = "Thanks for the feedback! Logged for training improvements."

    return render_template("index.html", feedback_msg=msg)


# ---------------------------------------------------
# RUN FLASK
# ---------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
