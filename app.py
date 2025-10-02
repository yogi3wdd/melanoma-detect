from fastapi import FastAPI, UploadFile
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Load model
model = tf.keras.models.load_model("melanoma_detect.h5")

# Label sesuai dataset (contoh: 0 = Normal, 1 = Melanoma)
class_names = ["Normal", "Melanoma"]

@app.post("/predict")
async def predict(file: UploadFile):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
    
    # Preprocess
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    prediction = model.predict(img_array)[0]  # ambil array hasil prediksi
    pred_idx = np.argmax(prediction)          # ambil kelas dengan probabilitas tertinggi
    confidence = float(np.max(prediction))    # tingkat akurasi (probabilitas)

    return {
        "hasil": class_names[pred_idx],
        "akurasi": round(confidence * 100, 2)  # persen
    }