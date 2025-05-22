from fastapi import APIRouter
from app.models.model import DataInput,PreprocessRequest
from app.service.data_collection_service import DataCollectionService
from app.service.preprocessdata_service import PreProcessData
from app.service.trainingmodel_service import TrainingModel

router = APIRouter()

@router.post("/")
async def collect_data(payload: DataInput):
    return DataCollectionService.data_collection_service.DataCollectionService.save_data_from_file(payload)

@router.post("/preprocess")
async def preprocess(req: PreprocessRequest):
    count = PreProcessData.preprocess_data(req)
    return {"status": "Preprocessing complete", "processed_count": count}

@router.post("/training")
async def training():
    result = TrainingModel.train_model()
    return {"status": "Training started", "details": result}