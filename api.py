from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import nltk
from nltk.tokenize import RegexpTokenizer
import os
import json
from contextlib import asynccontextmanager

# Configure NLTK data path for Render
NLTK_DIR = "/tmp/nltk_data"
os.makedirs(NLTK_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DIR)

# Create tokenizer that doesn't require punkt_tab
tokenizer = RegexpTokenizer(r'\w+')

# Global variables for model and class names
model = None
class_names = None

# Define lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model
    global model, class_names
    model_dir = "model_output"
    model_path = os.path.join(model_dir, "model_pipeline.pkl")
    class_names_path = os.path.join(model_dir, "class_names.pkl")
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(class_names_path, 'rb') as f:
            class_names = pickle.load(f)
        print("Model and class names loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
    
    yield
    
    # Shutdown: nothing to clean up

app = FastAPI(
    title="Azerbaijani Literature Genre Classifier",
    description="API for predicting genres of Azerbaijani literature texts",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the stemmer using RegexpTokenizer instead of word_tokenize
def azerbaijani_stemmer(text):
    # Common Azerbaijani suffixes
    suffixes = [
        "dir", "dır", "dur", "dür", 
        "miş", "mış", "muş", "müş", 
        "lar", "lər", "dan", "dən", 
        "a", "ə", "ın", "in", "un", "ün",
        "da", "də", "ki"
    ]
    
    # Common Azerbaijani stop words
    stop_words = [
        "bir", "və", "amma", "ki", "ilə", "bu", "o", "üçün", 
        "belə", "də", "artıq", "daha", "lakin", "çünki", "əgər"
    ]
    
    # Tokenize and lowercase using RegexpTokenizer
    words = tokenizer.tokenize(text.lower())
    
    # Remove stop words and apply stemming
    stemmed_words = []
    for word in words:
        # Skip stop words
        if word in stop_words:
            continue
            
        # Apply stemming
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                word = word[:-len(suffix)]
                break
        
        # Only add non-empty words
        if word and len(word) > 1:
            stemmed_words.append(word)
            
    return " ".join(stemmed_words)

# Input and output models
class TextInput(BaseModel):
    text: str = Field(..., description="Azerbaijani text to classify", min_length=10)
    minimum_confidence: float = Field(0.3, description="Minimum confidence threshold (0.0-1.0)", ge=0.0, le=1.0)

class PredictionResult(BaseModel):
    genre: str
    confidence: float
    all_predictions: dict

# Endpoint for genre prediction
@app.post("/predict/")
async def predict_genre(input_data: TextInput):
    global model, class_names
    
    # Define genre characteristics
    genre_characteristics = {
        "romantik": ["Emosional ekspressiya", "Lirik təsvir", "Fərdi hisslər", "Təbiət mənzərələri"],
        "fəlsəfi": ["Dərin düşüncələr", "Ekzistensial suallar", "Abstrakt ideyalar", "Mənəvi axtarış"],
        "tarixi": ["Tarixi hadisələr", "Real şəxsiyyətlər", "Dövrün təsviri", "Epik narrativ"],
        "lirik": ["Şəxsi duyğular", "Qısa və ritmik forma", "Musiqili ton", "Subyektiv baxış"],
        "epik": ["Təhkiyə", "Təsviri elementlər", "Qəhrəmanlıq motivləri", "Geniş məkan və zaman"],
        "satirik": ["İroni və yumor", "Sosial tənqid", "Mübaliğə", "Kəskin müşahidələr"]
    }
    
    # Check if model is loaded
    if model is None or class_names is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")
    
    # Preprocess the text
    preprocessed_text = azerbaijani_stemmer(input_data.text)
    
    # Make prediction
    try:
        # Get predicted probabilities
        probabilities = model.predict_proba([preprocessed_text])[0]
        
        # Map probabilities to genre names
        all_predictions = {class_name: float(prob) for class_name, prob in zip(class_names, probabilities)}
        
        # Find best prediction
        best_idx = probabilities.argmax()
        best_genre = class_names[best_idx]
        confidence = probabilities[best_idx]
        
        # Check confidence threshold
        if confidence < input_data.minimum_confidence:
            return {
                "genre": "unknown",
                "confidence": 0.0,
                "all_predictions": all_predictions,
                "characteristics": []  # Empty characteristics for unknown genre
            }
        
        return {
            "genre": best_genre,
            "confidence": f"{round(float(confidence) * 100, 2)}%",
            "characteristics": genre_characteristics.get(best_genre, []),
            "all_predictions": {k: f"{round(v * 100, 2)}%" for k, v in all_predictions.items()}
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Endpoint for checking service health
@app.get("/health/")
async def health_check():
    if model is None:
        return {"status": "warning", "message": "Model not loaded"}
    return {"status": "ok", "message": "Service is operational"}

# Add this handler for serverless deployment
from mangum import Mangum
handler = Mangum(app)

# Don't run the app when imported by serverless platform
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)