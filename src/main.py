"""
FastAPI application for Welli - Digital Wellness Assistant
Modular retention engine with ML-powered personalization
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from src.api.routes import router
import uvicorn
import os

# Load environment variables from .env file
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Welli - Digital Wellness Assistant",
    description="AI-powered wellness retention engine with goal matching, behavioral clustering, churn prediction, and micro-coaching",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Welli API is running",
        "version": "1.0.0",
        "endpoints": [
            "/api/v1/match-goal",
            "/api/v1/cluster-user", 
            "/api/v1/predict-churn",
            "/api/v1/daily-plan"
        ]
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {"status": "healthy", "service": "welli-api"}

if __name__ == "__main__":
    # Get configuration from environment variables
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "True").lower() == "true"
    
    # Verify OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("💡 Make sure your .env file contains: OPENAI_API_KEY=your_key_here")
        exit(1)
    
    print("🚀 Starting Welli API Server...")
    print(f"📍 Server: http://{host}:{port}")
    print(f"📖 API Docs: http://{host}:{port}/docs")
    
    uvicorn.run(app, host=host, port=port, reload=reload)
