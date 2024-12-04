from ultralytics import YOLO

class YoloPoolable:
    def __init__(self, yolo):
        self.yolo = yolo
        self.isReady = True

    def reset(self):
        self.isReady = True

    def __getattr__(self, name):
        return getattr(self.yolo, name)

    def __setattr__(self, name, value):
        if name.startswith('_'):  
            super().__setattr__(name, value)
        else:
            setattr(self.yolo, name, value)