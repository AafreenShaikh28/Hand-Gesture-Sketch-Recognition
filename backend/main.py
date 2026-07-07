from typing import List
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from prediction import preprocess_image, predict
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

class Guess(BaseModel):
    label: str
    confidence: float

class PredictionResponse(BaseModel):
    guesses: List[Guess]

@app.get("/")
def root():
    return {"message": "API is running!"}

@app.post("/predict", response_model=PredictionResponse)
async def get_prediction(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image_array = preprocess_image(image_bytes)
    guesses = predict(image_array)
    return PredictionResponse(guesses=guesses)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)