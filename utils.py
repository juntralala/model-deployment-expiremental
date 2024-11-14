import tensorflow as tf
from PIL import Image
import numpy as np


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

