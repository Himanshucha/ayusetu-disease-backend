import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # <-- ADDED
from pydantic import BaseModel, field_validator
import uvicorn

# --- PATH RESOLUTION FIX ---
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.predict import get_full_diagnosis

app = FastAPI(title="AyurPredict API")

# --- 1. ADD CORS MIDDLEWARE ---
# This allows your React frontend to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, you can replace "*" with your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATA SCHEMAS WITH VALIDATION ---
class SymptomPayload(BaseModel):
    symptoms: dict[str, float]

    @field_validator('symptoms')
    @classmethod
    def validate_levels(cls, v):
        for symptom, level in v.items():
            if level not in [1, 2, 3]:
                raise ValueError(f"Symptom '{symptom}' has level {level}. Only levels 1, 2, or 3 are allowed.")
        return v

@app.get("/")
def health_check():
    return {"status": "AyurPredict API is online"}

@app.post("/predict")
def predict(payload: SymptomPayload):
    try:
        result = get_full_diagnosis(payload.symptoms)
        
        # Check if the result returned an error from predict.py
        if "Error" in result:
            raise HTTPException(status_code=400, detail=result["Error"])
            
        return result
        
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# This part is for LOCAL testing. 
# Render will use its own Start Command (uvicorn src.api:app)
if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)