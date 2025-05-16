from fastapi import APIRouter, Query, HTTPException
from app.models.model import EvenNumbersResponse,AutoResponse,SimilarityResponse
from app.service.nlp_service import NlpService

router = APIRouter()

@router.get("/even", response_model=EvenNumbersResponse)
async def get_even_numbers(start: int = Query(..., ge=0),end: int = Query(..., ge=0)):
    if start > end:
        raise HTTPException(status_code=400, detail="Start must be less than or equal to end")  
    evens = [i for i in range(start, end + 1) if i % 2 == 0]
    return {"even_numbers": evens}

@router.get('/auto-response', response_model=AutoResponse)
async def auto_response(words:str):
    return await NlpService.auto_response(words)

@router.get('/similarity', response_model=SimilarityResponse)
async def similarity_score(text1:str,text2:str):
    return await NlpService.similarity_score(text1,text2)