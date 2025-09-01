from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
import os
import logging
import time  # Added missing import
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import your RAG engine
from rag_engine import RagEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TechCraft Chatbot API",
    version="1.0.0",
    description="RAG-based chatbot API for TechCraft Solutions",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware - Allow all origins for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pingus-frontend.vercel.app"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
BASE_DIR = Path(__file__).parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
UPLOAD_DIR = BASE_DIR / "uploads"
CHROMA_DB_DIR = BASE_DIR / "chroma_langchain_db"

# Create directories if they don't exist
for directory in [KNOWLEDGE_BASE_DIR, UPLOAD_DIR, CHROMA_DB_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Initialize RAG engine
try:
    rag_engine = RagEngine(
        doc_folder=str(KNOWLEDGE_BASE_DIR),
        persist_directory=str(CHROMA_DB_DIR)
    )
    logger.info("RAG engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG engine: {e}")
    rag_engine = None

# Pydantic models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[Message]] = []
    user_id: Optional[str] = None
    session_id: Optional[str] = None  # Added session_id field

class Source(BaseModel):
    title: str
    url: str
    content: str

class ChatResponse(BaseModel):
    response: str
    sources: List[Source]
    session_id: str
    success: bool

class FeedbackRequest(BaseModel):
    session_id: str
    message_id: str
    helpful: bool
    feedback: Optional[str] = None

class DocumentResponse(BaseModel):
    name: str
    size: int
    modified: float
    pages: Optional[int] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    active_sessions: int
    rag_engine_initialized: bool
    document_count: int
    environment: str

# In-memory storage (replace with database in production)
conversations = {}
feedback_data = {}

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "TechCraft RAG Chatbot API", "version": "1.0.0"}

# Fixed: Single /health endpoint (removed duplicate)
@app.get("/health")
async def health_check():
    """Health check endpoint for frontend connection testing"""
    try:
        document_count = len(list(KNOWLEDGE_BASE_DIR.glob("*.*"))) if KNOWLEDGE_BASE_DIR.exists() else 0
        
        return JSONResponse(
            content={
                "status": "healthy" if rag_engine else "degraded",
                "timestamp": time.time(),
                "service": "TechCraft RAG API",
                "active_sessions": len(conversations),
                "rag_engine_initialized": rag_engine is not None,
                "document_count": document_count,
                "environment": os.getenv("ENVIRONMENT", "development")
            }, 
            status_code=200
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            content={
                "status": "error",
                "timestamp": time.time(),
                "service": "TechCraft RAG API",
                "error": str(e)
            },
            status_code=500
        )

@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Send a message to the chatbot and get a response
    """
    try:
        if not rag_engine:
            raise HTTPException(status_code=503, detail="RAG engine not initialized")
        
        # Generate or retrieve session ID - Fixed to use session_id from request
        session_id = request.session_id or request.user_id or f"session_{uuid.uuid4().hex[:12]}"
        
        # Initialize session if it doesn't exist
        if session_id not in conversations:
            conversations[session_id] = {
                "created_at": datetime.now().isoformat(),
                "history": [],
                "message_count": 0
            }
        
        # Add user message to history
        user_message_id = f"msg_{uuid.uuid4().hex[:8]}"
        user_message = {
            "id": user_message_id,
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now().isoformat()
        }
        conversations[session_id]["history"].append(user_message)
        conversations[session_id]["message_count"] += 1
        
        # Convert history to format expected by RAG engine
        rag_history = [
            {"role": msg.role, "content": msg.content} 
            for msg in request.conversation_history
        ]
        
        # Process with RAG engine
        rag_result = rag_engine.query(request.message, rag_history)
        
        # Add assistant response to history
        assistant_message_id = f"msg_{uuid.uuid4().hex[:8]}"
        assistant_message = {
            "id": assistant_message_id,
            "role": "assistant",
            "content": rag_result["response"],
            "timestamp": datetime.now().isoformat(),
            "sources": rag_result.get("sources", [])
        }
        conversations[session_id]["history"].append(assistant_message)
        
        # Clean up old conversations in background
        background_tasks.add_task(cleanup_old_conversations)
        
        return ChatResponse(
            response=rag_result["response"],
            sources=[Source(**source) for source in rag_result.get("sources", [])],
            session_id=session_id,
            success=rag_result.get("success", True)
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversation/{session_id}", tags=["Conversation"])
async def get_conversation(session_id: str):
    """
    Get conversation history for a specific session
    """
    if session_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return conversations[session_id]

@app.delete("/api/conversation/{session_id}", tags=["Conversation"])
async def delete_conversation(session_id: str):
    """
    Delete a conversation session
    """
    if session_id in conversations:
        del conversations[session_id]
        return {"message": "Conversation deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Conversation not found")

@app.post("/api/feedback", tags=["Feedback"])
async def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback for a chatbot response
    """
    try:
        if request.session_id not in feedback_data:
            feedback_data[request.session_id] = {}
        
        feedback_data[request.session_id][request.message_id] = {
            "helpful": request.helpful,
            "feedback": request.feedback,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Received feedback for session {request.session_id}, message {request.message_id}")
        return {"message": "Feedback submitted successfully"}
    
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail="Error submitting feedback")

@app.post("/api/upload", tags=["Knowledge Base"])  # Fixed: Changed from /api/knowledge/upload to /api/upload
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document to the knowledge base
    """
    try:
        # Validate file type
        allowed_extensions = ['.docx', '.pdf', '.txt']
        if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
            raise HTTPException(status_code=400, detail="Only .docx, .pdf, and .txt files are supported")
        
        # Validate file size (max 10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        content = await file.read()
        if len(content) > max_size:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB")
        
        # Save the uploaded file to knowledge base
        file_path = KNOWLEDGE_BASE_DIR / file.filename
        
        # Check if file already exists
        if file_path.exists():
            raise HTTPException(status_code=400, detail="File with this name already exists")
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Add to RAG system if method exists
        if hasattr(rag_engine, 'add_document'):
            success, message = rag_engine.add_document(str(file_path))
            
            if success:
                logger.info(f"Document uploaded successfully: {file.filename}")
                return {"message": f"Document uploaded and processed: {message}"}
            else:
                # Clean up the file if processing failed
                if file_path.exists():
                    file_path.unlink()
                raise HTTPException(status_code=400, detail=message)
        else:
            # If add_document method doesn't exist, just save the file and reinitialize
            logger.info(f"Document uploaded successfully: {file.filename}")
            return {"message": f"Document uploaded successfully: {file.filename}"}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading document: {e}")

# Fixed: Updated /api/files to use KNOWLEDGE_BASE_DIR instead of hardcoded 'data' directory
@app.get("/api/files", response_model=List[DocumentResponse], tags=["Knowledge Base"])
async def list_documents():
    """
    Get list of uploaded files
    """
    try:
        files = []
        
        # Check if knowledge base directory exists
        if not KNOWLEDGE_BASE_DIR.exists():
            return []
        
        for file_path in KNOWLEDGE_BASE_DIR.glob("*.*"):
            if file_path.suffix.lower() in ['.docx', '.pdf', '.txt']:
                # Count pages in document (approximate for docx)
                pages = None
                if file_path.suffix.lower() == '.docx':
                    try:
                        from docx import Document as DocxDocument
                        doc = DocxDocument(file_path)
                        pages = len(doc.element.body) // 10  # Rough estimate
                    except:
                        pages = None
                
                files.append({
                    "name": file_path.name,
                    "size": file_path.stat().st_size,
                    "modified": file_path.stat().st_mtime,
                    "pages": pages,
                    "type": "document"
                })
        
        return files
        
    except Exception as e:
        logger.error(f"Error getting files: {e}")
        return []

# Fixed: Updated to use KNOWLEDGE_BASE_DIR
@app.delete("/api/files/{filename}", tags=["Knowledge Base"])
async def delete_file(filename: str):
    """Delete a file from the knowledge base"""
    try:
        # Security check to prevent path traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        file_path = KNOWLEDGE_BASE_DIR / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Remove file
        file_path.unlink()
        
        logger.info(f"File deleted: {filename}")
        return {"message": f"File {filename} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/refresh", tags=["Knowledge Base"])
async def refresh_knowledge_base_endpoint():
    """Refresh the knowledge base (reload all documents)"""
    try:
        # Reinitialize the RAG engine to refresh knowledge base
        global rag_engine
        rag_engine = RagEngine(
            doc_folder=str(KNOWLEDGE_BASE_DIR),
            persist_directory=str(CHROMA_DB_DIR)
        )
        
        logger.info("Knowledge base refreshed successfully")
        return {"message": "Knowledge base refreshed successfully"}
        
    except Exception as e:
        logger.error(f"Error refreshing knowledge base: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/version", tags=["System"])
async def get_version():
    """
    Get API version information
    """
    return {
        "name": "TechCraft RAG Chatbot API",
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "development")
    }

@app.get("/api/frontend/config", tags=["Frontend"])
async def get_frontend_config():
    """
    Get frontend configuration
    """
    return {
        "api_url": os.getenv("API_URL", "http://localhost:8000"),
        "websocket_url": os.getenv("WEBSOCKET_URL", "ws://localhost:8000"),
        "max_file_size": 10 * 1024 * 1024,
        "allowed_file_types": [".docx", ".pdf", ".txt"],
        "features": {
            "file_upload": True,
            "chat_history": True,
            "feedback": True,
            "document_management": True
        }
    }

def cleanup_old_conversations():
    """
    Clean up conversations older than 24 hours
    """
    try:
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session_data in conversations.items():
            created_at = datetime.fromisoformat(session_data["created_at"])
            if (current_time - created_at).total_seconds() > 24 * 3600:  # 24 hours
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del conversations[session_id]
            logger.info(f"Cleaned up expired session: {session_id}")
    
    except Exception as e:
        logger.error(f"Error cleaning up conversations: {e}")

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )
@app.get("/api/health")
async def api_health_check():
    """Health check endpoint for API path"""
    return await health_check()

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable or use default
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=os.getenv("ENVIRONMENT") == "development"
    )
