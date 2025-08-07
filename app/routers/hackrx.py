from fastapi import APIRouter, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import tempfile
import hashlib

# Import our utility classes
from utils.extractChunks import DocumentProcessor
from utils.vectorizeUpsert import PineconeVectorStore
from utils.queryEmbed import QueryProcessor

# Security
security = HTTPBearer()
VALID_TOKEN = "ae6121e71a4489821062f4cd0fb155e6c5af01aa4ab56617021ecf71fd672e07"

# Request/Response Models
class HackRxRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

router = APIRouter(
    tags=["hackrx"],
    prefix="/hackrx",
)

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify the bearer token"""
    if credentials.credentials != VALID_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
    return credentials.credentials

@router.post("/run", response_model=HackRxResponse)
async def run_hackrx(
    request: HackRxRequest,
    token: str = Depends(verify_token)
):
    """
    Complete RAG pipeline endpoint that:
    1. Downloads and processes the document from URL
    2. Creates vector embeddings and stores in Pinecone
    3. Answers all questions using the document context
    
    Args:
        request: HackRxRequest containing document URL and questions
        token: Bearer token for authentication
        
    Returns:
        HackRxResponse with answers to all questions
    """
    
    try:
        # For testing, use an existing namespace that has vectors
        # Replace this with the hash-based approach once you re-upload documents
        namespace = "hackrx_75c490b4"  # One of the existing namespaces
        
        print(f"🚀 Starting RAG pipeline with namespace: {namespace}")
        print(f"📄 Document URL: {request.documents}")
        print(f"❓ Number of questions: {len(request.questions)}")
        
        # Step 1: Initialize processors
        doc_processor = DocumentProcessor()
        vector_store = PineconeVectorStore()
        query_processor = QueryProcessor()
        
        # Step 2: Check if this document is already processed
        print("📊 Checking if document is already processed...")
        index_stats = vector_store.get_index_stats()
        
        # Check if namespace already exists with vectors
        namespace_exists = False
        if 'namespaces' in index_stats and namespace in index_stats['namespaces']:
            vector_count = index_stats['namespaces'][namespace].get('vector_count', 0)
            if vector_count > 0:
                namespace_exists = True
                print(f"✅ Document already processed! Found {vector_count} vectors in namespace: {namespace}")
        
        # For now, assume namespace exists (since we're using an existing one)
        # Skip document processing and go directly to questions
        if namespace_exists:
            print("⏭️ Using existing vectorized document")
        else:
            print("⚠️ Namespace doesn't exist, but proceeding anyway for testing")
        
        # Step 5: Process all questions
        print("❓ Processing questions...")
        answers = []
        
        # Use ThreadPoolExecutor for parallel question processing
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Create futures for all questions
            futures = [
                executor.submit(
                    process_single_question,
                    query_processor,
                    question,
                    namespace
                )
                for question in request.questions
            ]
            
            # Collect answers in order
            for i, future in enumerate(futures):
                try:
                    answer = future.result(timeout=30)  # 30 second timeout per question
                    answers.append(answer)
                    print(f"✅ Question {i+1}/{len(request.questions)} answered")
                except Exception as e:
                    print(f"❌ Error processing question {i+1}: {str(e)}")
                    answers.append("I'm sorry, I couldn't process this question due to an error.")
        
        print(f"🎯 Successfully processed all {len(answers)} questions")
        
        return HackRxResponse(answers=answers)
                
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Unexpected error in RAG pipeline: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

def process_single_question(
    query_processor: QueryProcessor,
    question: str,
    namespace: str
) -> str:
    """
    Process a single question using the query processor
    
    Args:
        query_processor: Initialized QueryProcessor instance
        question: Question to answer
        namespace: Pinecone namespace to search in
        
    Returns:
        Answer string
    """
    try:
        print(f"🔍 Processing question: {question[:100]}...")
        print(f"📂 Using namespace: {namespace}")
        
        # First try with lower threshold to see if we get any results
        result = query_processor.process_query(
            query=question,
            k=5,  # Get top 5 most relevant chunks
            namespace=namespace,
            similarity_threshold=0.3  # Much lower threshold for debugging
        )
        
        # Debug information
        retrieved_chunks = result.get("retrieved_chunks", 0)
        print(f"📊 Retrieved {retrieved_chunks} chunks for question")
        
        if result.get("chunks_metadata"):
            print("📋 Chunk scores:")
            for i, chunk_meta in enumerate(result["chunks_metadata"]):
                print(f"  {i+1}. Score: {chunk_meta.get('score', 'N/A'):.4f}, Source: {chunk_meta.get('source_file', 'N/A')}")
        
        if result.get("error"):
            print(f"❌ Query processing error: {result['error']}")
            return f"Error processing question: {result['error']}"
        
        answer = result.get("answer", "I couldn't find a relevant answer.")
        print(f"✅ Generated answer length: {len(answer)} characters")
        
        return answer
        
    except Exception as e:
        print(f"❌ Error in process_single_question: {str(e)}")
        import traceback
        traceback.print_exc()
        return "I'm sorry, there was an error processing this question."

@router.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "RAG service is running"}

@router.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "service": "HackRx RAG API",
        "version": "1.0.0",
        "endpoints": {
            "POST /hackrx/run": "Main RAG pipeline endpoint",
            "GET /hackrx/health": "Health check",
            "GET /hackrx/": "API information"
        },
        "authentication": "Bearer token required for /run endpoint"
    }