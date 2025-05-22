from pydantic import BaseModel, Field
from typing import List,Dict,Optional

class ListStringResponse(BaseModel):
    words: List[str]


class SimilarityResponse(BaseModel):
    similarity_score : float


class SentimentResponse(BaseModel):
    label: str
    score: float

class StringResponse(BaseModel):
    string: str

class ObjectResponse(BaseModel):
    entities: Dict[str,str]

class DataInput(BaseModel):
    train_file: str = "app/data/job_summarize.json"

class PreprocessRequest(BaseModel):
    tokenize: bool = True
    max_length: int = 512

class TrainRequest(BaseModel):
    train_file: str = "app/data/job_summarize.json"
    model_name: str = "google/pegasus-xsum"
    output_dir: str = "./custom-summarizer-model"
    num_train_epochs: Optional[int] = 3
    batch_size: Optional[int] = 2

    