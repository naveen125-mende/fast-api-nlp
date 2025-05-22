#app/main.py
from fastapi import FastAPI
from app.routes.nlp_routes import router as nlp_router
from app.routes.training_routes import router as training_router
app = FastAPI()

app.include_router(nlp_router,prefix="/api/nlp" , tags=["NLP Techinques"])
app.include_router(training_router,prefix="/api/training-model" , tags=["Traing Model"])