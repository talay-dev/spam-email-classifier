from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import uvicorn
import numpy as np

# Load all saved components
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("spam_vectorizer.pkl")
scaler = joblib.load("spam_scaler.pkl")
pca = joblib.load("spam_pca.pkl")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class EmailRequest(BaseModel):
    text: str

class EmailResponse(BaseModel):
    is_spam: bool
    probability: float

@app.post("/predict", response_model=EmailResponse)
def predict_spam(request: EmailRequest):
    # Transform the input text using the vectorizer
    text_vectorized = vectorizer.transform([request.text])
    
    # Convert to array and apply scaling
    text_array = text_vectorized.toarray()
    text_scaled = scaler.transform(text_array)
    
    # Apply PCA transformation
    text_pca = pca.transform(text_scaled)
    
    # Make prediction
    prediction = model.predict(text_pca)[0]
    probability = model.predict_proba(text_pca)[0][1]
    
    return EmailResponse(
        is_spam=bool(prediction),
        probability=float(probability)
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=20000)
