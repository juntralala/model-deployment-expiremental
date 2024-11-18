from pydantic import BaseModel

class Item(BaseModel):
    model: str
    