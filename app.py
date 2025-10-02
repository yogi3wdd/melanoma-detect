from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Init Flask
app = Flask(__name__)

# Load model H5
model = tf.keras.models.load_model("melanoma_detect.h5")

# Endpoint prediksi
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files["image"].read()
    image = Image.open(io.BytesIO(file)).resize((224,224))  # sesuaikan ukuran input model
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]
    label = "Melanoma" if pred > 0.5 else "Non-Melanoma"
    acc = float(pred if pred > 0.5 else (1 - pred))

    return jsonify({"result": label, "accuracy": acc})

# Run
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)