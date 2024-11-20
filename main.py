from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from utils import Model
import tensorflow as tf
from model import Item

app = FastAPI()
labels = ["Fresh", "Rotten"]
@app.get("/")
async def root():
    return {"message": "AI Model is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file.file.seek(0)
    img = Image.open(file.file)
    img = img.resize((224, 224))
    imageToBinary = img.convert('RGB')
    with tf.device('/cpu:0'):
        model = Model("Fruit Classification", "./model/cabbage.keras")
        output = model.predict(imageToBinary)
        model.reset()
    return {"prediction": labels[output.argmax()],
            "confidence": str(output.max())}
    

    