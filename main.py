from fastapi import FastAPI, File, UploadFile
from PIL import Image
from utils import Model, preprocessData, fruitModel
import tensorflow as tf
from ultralytics import YOLO


app = FastAPI()
labels = ["Fresh", "Rotten"]
@app.get("/")
async def root():
    return {"message": "AI Model is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    file.file.seek(0)
    img = Image.open(file.file)
    data=img.resize((640,640))
    yolo = YOLO("model/best.pt")
    res = yolo.predict(data, device='cpu')
    cropImg, label = preprocessData(data, res)
    
    with tf.device('/cpu:0'):
        model = Model("Fruit Classification", fruitModel(yolo.names[label]))
        output = model.predict(cropImg)
        model.reset()
    return {"prediction": labels[output.argmax()] + " " +yolo.names[label],
            "confidence": str(output.max())}
    

    