from ultralytics import YOLO
import YoloPoolable
import asyncio

class YoloPool:
    def __init__(self, count = 2, timeout = 30):
        self.pool = []
        self.initPool(count)
        self.timeout = timeout
    
    def initPool(self, count: int):
        for i in range(count):
            self.pool.append(YoloPoolable(YOLO("model/best.pt")))
    
    def getYolo(self):
        self.timeOut()
        while(True):
            for yoloPoolable in self.pool:
                if(yoloPoolable.isReady):
                    yoloPoolable.isReady = False
                    return yoloPoolable

    async def timeOut(self):
        await asyncio.sleep(self.timeout)
        raise Exception('Time Out')