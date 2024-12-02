import tensorflow as tf
from PIL import Image
from numba import cuda 
import gc
import json
from model import Validation    


class Model():
    def __init__(self, model_name: str, model_path: str):
        self.model_name = model_name
        self.model_path = model_path
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, data: Image):
        data = tf.keras.preprocessing.image.img_to_array(data)
        data = tf.expand_dims(data, axis=0)
        # data = tf.keras.applications.mobilenet.preprocess_input(data)
        prediction = self.model.predict(data)
        return prediction
    def reset(self):
        #Check use gpu or no
        if tf.test.gpu_device_name():
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
            device = cuda.get_current_device()
            device.reset()
        else:
            del self.model
            tf.keras.backend.clear_session()
            gc.collect()
            self.model = None

def preprocessData(data: Image, res: list) -> tuple[Image.Image, str]: 
    box_coordinates = list(map(int, res[0].boxes.xyxy[0].tolist()))
    data = data.crop((box_coordinates[0], box_coordinates[1], box_coordinates[2], box_coordinates[3]))  
    data = data.resize((224, 224))
    data = data.convert('RGB')
    label = res[0].boxes.cls[0].item()
    return (data, label)
        
def fruitModel(label: str) -> tf.keras.models.Model:
    print(label)
    
    if label == 'apple':
        path = 'model/Apple.keras'
    elif label == 'banana':
        path = 'model/banana.keras'
    elif label == 'capsicum':
        path = 'model/capsicum.keras'
    elif label == 'tomato':
        path = 'model/tomato.keras'
    elif label == 'cabbage':
        path = 'model/cabbage.keras'
    else:
        return 'No model found'
    return path
def calculate_freshness(smell: str, texture:str, confidence:float, shop_verified: bool) -> float:
    smell_scores = {"fresh": 100, "neutral": 50, "rotten": 0}
    texture_scores = {"hard": 100, "normal": 50, "soft": 0}
    smell_score = smell_scores.get(smell, 50)  
    texture_score = texture_scores.get(texture, 50)  

    if shop_verified:
        shop_score = 100
    else:
        shop_score = 0
    
    
    total_score = float((smell_score + texture_score + confidence + shop_score) / 4)
    
    return total_score