#!/usr/bin/env python3
"""
Startup script for the Fracture Risk Prediction API.

This script ensures the API runs from the correct directory with all
necessary configurations and dependencies properly loaded.
"""

import os
import sys
from pathlib import Path

# Change to the project directory
project_dir = Path(__file__).parent
os.chdir(project_dir)

# Add src to Python path for imports
sys.path.insert(0, str(project_dir / "src"))

# Import and run the FastAPI app
if __name__ == "__main__":
    from src.api.main import app
    import uvicorn
    
    print("🚀 Starting Fracture Risk Prediction API...")
    print(f"📂 Working directory: {os.getcwd()}")
    print("🌐 Server will be available at: http://localhost:8000")
    print("📖 API documentation at: http://localhost:8000/docs")
    print("=" * 60)
    
    # Start the server - LOCALHOST ONLY (most secure)
    uvicorn.run(
        app,
        host="127.0.0.1",  # Only accessible from your computer
        port=8000,
        reload=False
    )
