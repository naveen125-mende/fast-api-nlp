from pydantic import BaseModel, Field
from typing import List

class EvenNumbersResponse(BaseModel):
    even_numbers: List[int]

class AutoResponse(BaseModel):
    words: List[str]

class SimilarityResponse(BaseModel):
    similarity_score : float