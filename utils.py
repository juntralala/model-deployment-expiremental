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
        

