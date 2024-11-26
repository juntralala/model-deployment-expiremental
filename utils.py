import tensorflow as tf
from PIL import Image
from numba import cuda 
import gc


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
