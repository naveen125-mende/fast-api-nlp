from pydantic import BaseModel, Field
from typing import List,Dict

class ListStringResponse(BaseModel):
    words: List[str]

class SimilarityResponse(BaseModel):
    similarity_score : float

class StringResponse(BaseModel):
    string: str

class ObjectResponse(BaseModel):
    entities: Dict[str,str]
    