from pydantic import BaseModel, EmailStr
from datetime import datetime
from pydantic import ConfigDict  

class EmailCreate(BaseModel):
    recipient: EmailStr
    subject: str
    body: str

class EmailOut(BaseModel):
    id: int
    sender: EmailStr
    recipient: EmailStr
    subject: str
    body: str
    timestamp: datetime
    read: bool

    model_config = ConfigDict(from_attributes=True) 

