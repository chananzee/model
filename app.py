from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# ==============================================================================
# ท่าไม้ตาย: แทรกแซงระบบ Core ของ Keras เพื่อดักลบตัวแปรที่มีปัญหาทิ้งทุก Layer
original_layer_init = tf.keras.layers.Layer.__init__

def safe_layer_init(self, *args, **kwargs):
    # หากมี quantization_config ติดมา ให้ลบทิ้งทันทีก่อนที่ Keras จะทันได้อ่าน
    kwargs.pop('quantization_config', None)
    original_layer_init(self, *args, **kwargs)

# นำฟังก์ชันที่เราปรับแต่งไปสวมทับฟังก์ชันเดิมของ Keras
tf.keras.layers.Layer.__init__ = safe_layer_init
# ==============================================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading model...")
# ตอนนี้โหลดแบบปกติได้เลย เพราะเราดัก Error ไว้ที่ต้นทางแล้ว
model = tf.keras.models.load_model('final_model.keras', compile=False)
print("Model loaded successfully! 🎉")

# กำหนดชื่อโรคตามคลาสที่คุณเทรนไว้
CLASS_NAMES = ["โรคใบจุด", "ใบปกติ", "ไวรัสโมสาก"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    image = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = round(100 * np.max(predictions[0]), 2)

    return {"disease": predicted_class, "confidence": confidence}