from pydantic import BaseModel

class Validation(BaseModel):
    smell: str | None
    texture: str | None
    verifiedShop: bool | None