from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import uvicorn

# =============================
# Konfigurasi Aplikasi & Model
# =============================
app = FastAPI(
    title="Melanoma Detection API",
    description="API untuk mendeteksi melanoma menggunakan model EfficientNetB0",
    version="1.0"
)

MODEL_PATH = "melanoma_detect.h5"
TARGET_SIZE = (128, 128)
CLASS_NAMES = ["normal", "melanoma"]  # urutan sesuai pelatihan kamu

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# =============================
# Fungsi bantu preprocessing
# =============================
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(TARGET_SIZE)
    arr = np.array(img).astype("float32") / 255.0  # sama seperti training
    arr = np.expand_dims(arr, axis=0)
    return arr

# =============================
# Endpoint utama untuk test di browser
# =============================
@app.get("/", response_class=HTMLResponse)
def main_page():
    return """
    <html>
        <head>
            <title>Melanoma Detector</title>
            <style>
                body { font-family: Arial; text-align: center; padding: 50px; background: #f0f0f0; }
                form { background: #fff; padding: 30px; border-radius: 10px; display: inline-block;
                       box-shadow: 0 0 10px rgba(0,0,0,0.1); }
                input[type=file] { margin: 10px 0; }
                input[type=submit] { background-color: #007BFF; color: white; border: none;
                                     padding: 10px 20px; border-radius: 5px; cursor: pointer; }
                input[type=submit]:hover { background-color: #0056b3; }
            </style>
        </head>
        <body>
            <h1>🔬 Melanoma Detection</h1>
            <p>Upload gambar kulit untuk mendeteksi apakah melanoma atau normal.</p>
            <form action="/predict_browser" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required><br>
                <input type="submit" value="Detect">
            </form>
        </body>
    </html>
    """

# =============================
# Endpoint prediksi via browser
# =============================
@app.post("/predict_browser", response_class=HTMLResponse)
async def predict_browser(file: UploadFile = File(...)):
    try:
        # Baca file upload (bytes)
        contents = await file.read()

        # Preprocessing untuk model
        img_arr = preprocess_image(contents)
        pred = model.predict(img_arr)[0][0]
        if pred < 0.5:
            label = "melanoma"
            confidence = 1 - pred  # karena semakin kecil pred → semakin yakin melanoma
        else:
            label = "normal"
            confidence = pred

        # Encode gambar ke base64 agar bisa ditampilkan di web
        import base64
        encoded = base64.b64encode(contents).decode('utf-8')

        # Warna hasil
        color = "red" if label == "melanoma" else "green"

        # HTML hasil prediksi + gambar
        return f"""
        <html>
        <body style='text-align:center;font-family:Arial;background:#f0f0f0;padding:40px'>
            <h1>🔍 Hasil Deteksi</h1>
            <img src="data:image/jpeg;base64,{encoded}" 
                 style="max-width:300px;border-radius:10px;margin:15px 0;">
            <h2 style='color:{color}'>{label.upper()}</h2>
            <p>Akurasi: {round(confidence * 100, 2)}%</p>
            <a href="/" style="text-decoration:none;color:#007BFF;">⬅️ Kembali</a>
        </body>
        </html>
        """
    except Exception as e:
        return HTMLResponse(f"<h3>Error: {str(e)}</h3>")

# =============================
# Endpoint untuk ESP32-CAM
# =============================
@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        x = preprocess_image(contents)
        preds = model.predict(x)
        pred = float(preds[0][0])  # ubah langsung ke float
        if pred < 0.5:
            label = "melanoma"
            confidence = 1 - pred  # karena semakin kecil pred → semakin yakin melanoma
        else:
            label = "normal"
            confidence = pred

        return JSONResponse({
            "label": str(label),
            "confidence": float(round(confidence * 100, 2))
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# =============================
# Jalankan server lokal
# =============================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
