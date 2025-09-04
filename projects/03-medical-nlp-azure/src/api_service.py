"""
Medical NLP API Service
FastAPI-based REST API for medical text analysis.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import sys
import os
from pathlib import Path
import logging
import uvicorn
from datetime import datetime

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir
shared_dir = current_dir.parent.parent / 'shared'

sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(shared_dir))

# Import our NLP pipeline
from nlp_pipeline import MedicalNLPPipeline

# Initialize FastAPI app
app = FastAPI(
    title="Medical NLP API",
    description="Advanced medical text analysis with Azure integration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline = None

# Request/Response Models
class TextAnalysisRequest(BaseModel):
    text: str = Field(..., description="Medical text to analyze", min_length=1, max_length=10000)
    use_azure: bool = Field(default=False, description="Whether to use Azure Text Analytics")
    include_preprocessing: bool = Field(default=True, description="Include preprocessing details")
    include_entities: bool = Field(default=True, description="Include entity extraction")
    include_classification: bool = Field(default=True, description="Include text classification")

class EntityResult(BaseModel):
    text: str
    category: str
    subcategory: Optional[str] = None
    confidence: float
    offset: int
    length: int
    normalized_text: Optional[str] = None

class PredictionResult(BaseModel):
    prediction: str
    confidence: float
    all_probabilities: Optional[Dict[str, float]] = None

class TextAnalysisResponse(BaseModel):
    original_text: str
    cleaned_text: Optional[str] = None
    expanded_text: Optional[str] = None
    tokens: Optional[List[str]] = None
    entities: Optional[List[EntityResult]] = None
    predictions: Optional[Dict[str, PredictionResult]] = None
    processing_time_ms: float
    timestamp: str

class BatchAnalysisRequest(BaseModel):
    texts: List[str] = Field(..., description="List of medical texts to analyze", max_items=100)
    use_azure: bool = Field(default=False, description="Whether to use Azure Text Analytics")
    include_details: bool = Field(default=False, description="Include detailed analysis for each text")

class BatchAnalysisResponse(BaseModel):
    total_texts: int
    results: List[TextAnalysisResponse]
    processing_time_ms: float
    timestamp: str

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str
    pipeline_ready: bool
    models_loaded: Dict[str, bool]
    azure_available: bool

class TrainingRequest(BaseModel):
    data_path: str = Field(..., description="Path to training data CSV file")
    config_overrides: Optional[Dict] = Field(default=None, description="Configuration overrides")

class TrainingResponse(BaseModel):
    status: str
    message: str
    results: Optional[Dict] = None
    timestamp: str

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the NLP pipeline on startup."""
    global pipeline
    
    try:
        # Initialize pipeline
        pipeline = MedicalNLPPipeline()
        
        # Try to load existing models
        try:
            pipeline.load_models('models')
            logging.info("Existing models loaded successfully")
        except:
            logging.warning("No existing models found. Training may be required.")
        
        logging.info("Medical NLP API started successfully")
        
    except Exception as e:
        logging.error(f"Failed to initialize pipeline: {str(e)}")
        raise

# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Check the health status of the API and pipeline."""
    
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        pipeline_ready=True,
        models_loaded=pipeline.models_trained,
        azure_available=pipeline.entity_recognizer.azure_ner.available
    )

# Single text analysis endpoint
@app.post("/analyze", response_model=TextAnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    """Analyze a single medical text."""
    
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        start_time = datetime.now()
        
        # Analyze the text
        result = pipeline.analyze_single_text(request.text)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Build response
        response_data = {
            "original_text": result["original_text"],
            "processing_time_ms": processing_time,
            "timestamp": result["timestamp"]
        }
        
        # Add preprocessing details if requested
        if request.include_preprocessing:
            response_data.update({
                "cleaned_text": result.get("cleaned_text"),
                "expanded_text": result.get("expanded_text"),
                "tokens": result.get("tokens", [])
            })
        
        # Add entities if requested
        if request.include_entities:
            entities = [
                EntityResult(
                    text=e["text"],
                    category=e["category"],
                    subcategory=e.get("subcategory"),
                    confidence=e["confidence"],
                    offset=0,  # Would need to calculate from original implementation
                    length=len(e["text"]),
                    normalized_text=e.get("normalized_text")
                )
                for e in result.get("entities", [])
            ]
            response_data["entities"] = entities
        
        # Add classification predictions if requested
        if request.include_classification and "predictions" in result:
            predictions = {}
            for pred_type, pred_data in result["predictions"].items():
                predictions[pred_type] = PredictionResult(
                    prediction=pred_data["prediction"],
                    confidence=pred_data["confidence"],
                    all_probabilities=pred_data.get("all_probabilities")
                )
            response_data["predictions"] = predictions
        
        return TextAnalysisResponse(**response_data)
        
    except Exception as e:
        logging.error(f"Error analyzing text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Batch analysis endpoint
@app.post("/analyze/batch", response_model=BatchAnalysisResponse)
async def analyze_batch(request: BatchAnalysisRequest):
    """Analyze multiple medical texts in batch."""
    
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    if len(request.texts) > 100:
        raise HTTPException(status_code=400, detail="Too many texts. Maximum 100 texts per batch.")
    
    try:
        start_time = datetime.now()
        
        results = []
        
        for text in request.texts:
            # Create individual request
            individual_request = TextAnalysisRequest(
                text=text,
                use_azure=request.use_azure,
                include_preprocessing=request.include_details,
                include_entities=request.include_details,
                include_classification=request.include_details
            )
            
            # Analyze each text
            result = await analyze_text(individual_request)
            results.append(result)
        
        # Calculate total processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return BatchAnalysisResponse(
            total_texts=len(request.texts),
            results=results,
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logging.error(f"Error in batch analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

# Entity extraction endpoint
@app.post("/entities")
async def extract_entities(text: str, use_azure: bool = False):
    """Extract medical entities from text."""
    
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        entities = pipeline.entity_recognizer.extract_entities(text, use_azure=use_azure)
        
        return {
            "text": text,
            "entities": [
                {
                    "text": e.text,
                    "category": e.category,
                    "subcategory": e.subcategory,
                    "confidence": e.confidence,
                    "offset": e.offset,
                    "length": e.length,
                    "normalized_text": e.normalized_text
                }
                for e in entities
            ],
            "total_entities": len(entities),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error extracting entities: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Entity extraction failed: {str(e)}")

# Classification endpoint
@app.post("/classify")
async def classify_text(text: str, task: str = "note_type"):
    """Classify medical text."""
    
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    if task not in ["note_type", "sentiment"]:
        raise HTTPException(status_code=400, detail="Task must be 'note_type' or 'sentiment'")
    
    try:
        if task == "note_type":
            if not pipeline.models_trained.get('note_classification', False):
                raise HTTPException(status_code=503, detail="Note classification model not trained")
            
            prediction = pipeline.note_classifier.predict_with_confidence([text])
            
        elif task == "sentiment":
            if not pipeline.models_trained.get('sentiment_analysis', False):
                raise HTTPException(status_code=503, detail="Sentiment analysis model not trained")
            
            prediction = pipeline.sentiment_analyzer.predict_with_confidence([text])
        
        return {
            "text": text,
            "task": task,
            "prediction": prediction[0]["prediction"],
            "confidence": prediction[0]["confidence"],
            "all_probabilities": prediction[0].get("all_probabilities", {}),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error in classification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

# Training endpoint
@app.post("/train", response_model=TrainingResponse)
async def train_models(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train or retrain the NLP models."""
    
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    if not os.path.exists(request.data_path):
        raise HTTPException(status_code=400, detail=f"Data file not found: {request.data_path}")
    
    try:
        # Load training data
        df = pipeline.load_data(request.data_path)
        
        # Apply configuration overrides if provided
        if request.config_overrides:
            pipeline.config.update(request.config_overrides)
        
        # Train models
        results = pipeline.train_classification_models(df)
        
        # Save models
        pipeline.save_models()
        
        return TrainingResponse(
            status="success",
            message=f"Models trained successfully on {len(df)} records",
            results=results,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logging.error(f"Error in training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

# Model info endpoint
@app.get("/models")
async def get_model_info():
    """Get information about loaded models."""
    
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return {
        "models_trained": pipeline.models_trained,
        "is_trained": pipeline.is_trained,
        "azure_available": pipeline.entity_recognizer.azure_ner.available,
        "config": pipeline.config,
        "timestamp": datetime.now().isoformat()
    }

# Configuration endpoint
@app.get("/config")
async def get_config():
    """Get current pipeline configuration."""
    
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return {
        "config": pipeline.config,
        "timestamp": datetime.now().isoformat()
    }

@app.put("/config")
async def update_config(config_updates: Dict):
    """Update pipeline configuration."""
    
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        pipeline.config.update(config_updates)
        
        return {
            "status": "success",
            "message": "Configuration updated successfully",
            "updated_config": pipeline.config,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error updating config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Config update failed: {str(e)}")

# Demo endpoint
@app.get("/demo")
async def demo_analysis():
    """Demonstrate the API with sample medical texts."""
    
    demo_texts = [
        "Patient presents with chest pain and shortness of breath. BP 140/90 mmHg, HR 85 bpm.",
        "The staff was excellent and very professional. Great experience!",
        "DISCHARGE SUMMARY: Patient discharged home in stable condition.",
        "Day 3: Patient feeling better. Continue current medications."
    ]
    
    results = []
    
    for text in demo_texts:
        request = TextAnalysisRequest(
            text=text,
            include_preprocessing=True,
            include_entities=True,
            include_classification=True
        )
        
        result = await analyze_text(request)
        results.append(result)
    
    return {
        "demo_results": results,
        "total_samples": len(demo_texts),
        "timestamp": datetime.now().isoformat()
    }

# Main function for running the server
def main():
    """Run the FastAPI server."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the server
    uvicorn.run(
        "api_service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
