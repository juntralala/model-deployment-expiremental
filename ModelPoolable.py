import tensorflow as tf
from numba import cuda 
from utils import Model

class ModelPoolable(Model):
    def __init__(self, model_name, model_path):
        super().__init__(model_name, model_path)
        self.isReady = True

    def reset(self):
        #Check use gpu or no
        if tf.test.gpu_device_name():
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
            device = cuda.get_current_device()
            device.reset()
        else:
            self.isReady = True