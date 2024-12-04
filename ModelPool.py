from ModelPoolable import ModelPoolable
import asyncio

class ModelPool:
    def __init__(self, count = 1, timeout = 30):
        self.pools = {
            'model/Apple.keras': [],
            'model/banana.keras': [],
            'model/capsicum.keras': [],
            'model/tomato.keras': [],
            'model/cabbage.keras': []}
        
        self.initPool(count)
        self.timeout = timeout
    
    def initPool(self, count: int):
        for key in self.pools:
            for i in range(count):
                self.pools[key].append(ModelPoolable("Fruit Classification", key))
    
    def getModel(self, path):
        pool = self.pools[path]
        self.timeOut()
        while(True):
            for modelPoolable in pool:
                if(modelPoolable.isReady):
                    modelPoolable.isReady = False
                    return modelPoolable

    async def timeOut(self):
        await asyncio.sleep(self.timeout)
        raise Exception('Time Out')