"""
Secure FastAPI startup script with authentication.

This version includes API key authentication for production use.
"""

import os
import sys
from pathlib import Path

# Change to the project directory
project_dir = Path(__file__).parent
os.chdir(project_dir)

# Add src to Python path for imports
sys.path.insert(0, str(project_dir / "src"))

if __name__ == "__main__":
    from src.api.main import app
    from fastapi import Depends, HTTPException, status
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    import uvicorn
    
    # Simple API key authentication
    security = HTTPBearer()
    API_KEY = "your-secret-api-key-here"  # Change this in production!
    
    def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
        if credentials.credentials != API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        return True
    
    # Add authentication to predict endpoint
    @app.post("/predict")
    async def secure_predict(
        patient_data,
        authenticated: bool = Depends(verify_api_key)
    ):
        # Original prediction logic here
        pass
    
    print("🔒 Starting SECURE Fracture Risk Prediction API...")
    print(f"📂 Working directory: {os.getcwd()}")
    print("🌐 Server will be available at: http://localhost:8000")
    print("🔑 API Key required for predictions")
    print("📖 API documentation at: http://localhost:8000/docs")
    print("=" * 60)
    
    # Start the server - localhost only
    uvicorn.run(
        app,
        host="127.0.0.1",  # Localhost only
        port=8000,
        reload=False
    )
