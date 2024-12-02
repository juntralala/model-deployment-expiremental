from fastapi import  FastAPI, File, Form, UploadFile
from PIL import Image
from utils import Model, preprocessData, fruitModel, calculate_freshness
import tensorflow as tf
from ultralytics import YOLO
from model import Validation
import json


app = FastAPI()
labels = ["Fresh", "Rotten"]
@app.get("/")
async def root():
    return {"message": "AI Model is running"}

@app.post("/predict")
async def predict(validation: str = Form(...) , file: UploadFile = File(...)):
    try:
        validation_data = json.loads(validation)
        validation_obj = Validation(**validation_data)  
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format for validation field"}
    file.file.seek(0)
    try:
        img = Image.open(file.file)
    except:
        return {"error": "Invalid image file"}
    data=img.resize((640,640))
    yolo = YOLO("model/best.pt")
    res = yolo.predict(data, device='cpu')
    cropImg, label = preprocessData(data, res)
    
    with tf.device('/cpu:0'):
        model = Model("Fruit Classification", fruitModel(yolo.names[label]))
        output = model.predict(cropImg)
        model.reset()

    if label == 0:
        freshness = calculate_freshness(validation_obj.smell, validation_obj.texture, output.max(), validation_obj.verifiedShop)
    else:
        freshness = calculate_freshness(validation_obj.smell, validation_obj.texture, 100 - output.max(), validation_obj.verifiedShop)
        
    
    return {"prediction": labels[output.argmax()] + " " +yolo.names[label],
            "confidence": freshness}
    

    