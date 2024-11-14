from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from utils import Model
import os

model = Model("Fruit Classification", "./model/classifiers.keras")
labels = ['freshcabbage', 'freshapples', 'freshbanana', 'freshcapsicum',   'freshtomato', 'rottencabbage','rottenapples', 'rottenbanana', 'rottencapsicum',  'rottentomato']
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file.file.seek(0)
    img = Image.open(file.file)
    img = img.resize((224, 224))
    imageToBinary = img.convert('RGB')
    output = model.predict(imageToBinary)   
    return {"prediction": labels[output.argmax()],
            "confidence": str(output.max())}
    

    