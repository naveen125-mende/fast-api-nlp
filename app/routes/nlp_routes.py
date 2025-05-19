from fastapi import APIRouter, Query, HTTPException
from app.models.model import ListStringResponse,SimilarityResponse,StringResponse,ObjectResponse
from app.service.nlp_service import NlpService

router = APIRouter()

@router.get("/even", response_model=ListStringResponse)
async def get_even_numbers(start: int = Query(..., ge=0),end: int = Query(..., ge=0)):
    if start > end:
        raise HTTPException(status_code=400, detail="Start must be less than or equal to end")  
    evens = [i for i in range(start, end + 1) if i % 2 == 0]
    return {"even_numbers": evens}

@router.get('/auto-response', response_model=ListStringResponse)
async def auto_response(words:str):
    return await NlpService.auto_response(words)

@router.get('/similarity', response_model=SimilarityResponse)
async def similarity_score(text1:str,text2:str):
    return await NlpService.similarity_score(text1,text2)

@router.get('/behaviour', response_model=StringResponse)
async def behaviour_of_sentence(sentence:str):
    return await NlpService.behaviour_of_sentence(sentence)

@router.get('/entities', response_model=ObjectResponse)
async def extract_entities(sentence:str):
    return await NlpService.extract_entities(sentence)

@router.get('/text-summarize', response_model=StringResponse)
async def text_summarize(sentence:str):
    return await NlpService.text_summarize(sentence)

@router.get('/keywords', response_model=ListStringResponse)
async def extract_keywords(sentence:str):
    return await NlpService.extract_keywords(sentence)