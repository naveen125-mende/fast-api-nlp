#app/main.py
from fastapi import FastAPI
from app.routes.nlp_routes import router as nlp_router

app = FastAPI()

app.include_router(nlp_router,prefix="/api/nlp" , tags=["NLP Techinques"])