from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load model
model = tf.keras.models.load_model("melanoma_detect.h5")

app = Flask(__name__)

def predict(img):
    img = img.resize((224,224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    pred = model.predict(img_array)[0][0]

    confidence = float(pred)*100 if pred >= 0.5 else (1-float(pred))*100
    label = "Melanoma" if pred >= 0.5 else "Bukan Melanoma"
    return label, confidence

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    label, confidence = predict(img)

    return jsonify({
        "result": label,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)