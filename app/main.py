from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.hackrx import router as hackrx_router

app = FastAPI(
    title="HackRx RAG API",
    description="Document processing and question answering API using RAG pipeline",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers with API versioning
app.include_router(hackrx_router, prefix="/api/v1")

@app.get("/")
def read_root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "Welcome to the HackRX RAG API!",
        "version": "1.0.0",
        "docs": "/docs",
        "api_base": "/api/v1",
        "endpoints": {
            "POST /api/v1/hackrx/run": "Main RAG pipeline endpoint",
            "GET /api/v1/hackrx/health": "Health check",
        }
    }

@app.get("/health")
def health_check():
    """
    Global health check endpoint.
    """
    return {"status": "healthy", "service": "HackRx RAG API"}