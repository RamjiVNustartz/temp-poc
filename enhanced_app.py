# AI Document Intelligence Assistant - FastAPI Backend


from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import base64
import requests
import io
from PIL import Image
from dotenv import load_dotenv
import os
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="DocuMind AI - Document Intelligence Assistant")

# Template configuration
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# API Configuration
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the .env file")

# Store conversation sessions
conversation_sessions = {}

# System prompts for document analysis
DOCUMENT_ANALYSIS_PROMPT = """You are DocuMind AI, an advanced document intelligence assistant specializing in comprehensive document analysis.

Your capabilities include:
- Text extraction and OCR
- Document summarization
- Key information extraction
- Sentiment and tone analysis
- Entity recognition (people, organizations, dates, locations)
- Action item identification
- Insight generation

**Task**: Analyze the provided document image and respond to the user's query: {query}

**Instructions**:
1. **Extract & Understand**: Carefully read all visible text in the document
2. **Structure Your Response**: Organize information clearly with headers and bullet points
3. **Be Comprehensive**: Cover all relevant aspects of the query
4. **Highlight Key Points**: Use **bold** for important information
5. **Provide Context**: Explain significance where helpful

**Response Format**:
## 📄 Document Overview
[Brief description of document type and purpose]

## 🔑 Key Findings
[Main points, facts, or answers to the query]

## 💡 Insights & Analysis
[Deeper analysis, patterns, or important observations]

## 📋 Summary
[Concise summary of the most important information]

Now analyze the document and answer: {query}"""

FOLLOWUP_PROMPT = """You are DocuMind AI in a follow-up conversation about a previously analyzed document.

**Instructions**:
- Answer the user's question directly and concisely
- Reference information from the document when relevant
- Keep responses focused (3-5 sentences unless more detail is needed)
- Maintain context from the previous analysis
- Be helpful and professional

Respond to: {query}"""

def call_groq_api(messages: list, max_tokens: int = 2000) -> requests.Response:
    """Call Groq API with Llama 4 Maverick model"""
    try:
        response = requests.post(
            GROQ_API_URL,
            json={
                "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.95
            },
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            timeout=60
        )
        return response
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Request timeout. Please try again.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API request failed: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main application page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze_document")
async def analyze_document(
    document: UploadFile = File(...),
    query: str = Form(...),
    session_id: str = Form(...)
):
    """Analyze uploaded document and answer user's query"""
    try:
        # Read and validate document
        doc_content = await document.read()
        if not doc_content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Encode document image
        encoded_doc = base64.b64encode(doc_content).decode("utf-8")
        
        # Verify image format
        try:
            img = Image.open(io.BytesIO(doc_content))
            img.verify()
        except Exception as e:
            logger.error(f"Invalid image format: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid file format: {str(e)}")
        
        # Prepare analysis prompt
        enhanced_query = DOCUMENT_ANALYSIS_PROMPT.format(query=query)
        
        # Initialize conversation
        messages = [
            {
                "role": "system",
                "content": "You are DocuMind AI, an expert document analysis assistant. Provide clear, structured, and insightful responses."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": enhanced_query},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_doc}"}}
                ]
            }
        ]
        
        logger.info(f"Analyzing document for session: {session_id}")
        response = call_groq_api(messages)
        
        if response.status_code == 200:
            result = response.json()
            assistant_message = result["choices"][0]["message"]["content"]
            
            # Store session with follow-up context
            conversation_sessions[session_id] = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are DocuMind AI. Provide concise, helpful answers based on the analyzed document."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Document analysis request: {query}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_doc}"}}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": assistant_message
                    }
                ],
                "document": encoded_doc
            }
            
            logger.info("Document analysis completed successfully")
            return JSONResponse(status_code=200, content={
                "response": assistant_message,
                "status": "success"
            })
        else:
            logger.error(f"API error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=response.status_code, detail="Analysis failed")
    
    except HTTPException as he:
        logger.error(f"HTTP Exception: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/continue_conversation")
async def continue_conversation(
    query: str = Form(...),
    session_id: str = Form(...)
):
    """Handle follow-up questions about the document"""
    try:
        # Check session exists
        if session_id not in conversation_sessions:
            raise HTTPException(status_code=404, detail="Session not found. Please upload a document first.")
        
        session = conversation_sessions[session_id]
        
        # Add follow-up query
        session["messages"].append({
            "role": "user",
            "content": FOLLOWUP_PROMPT.format(query=query)
        })
        
        logger.info(f"Processing follow-up for session: {session_id}")
        response = call_groq_api(session["messages"], max_tokens=800)
        
        if response.status_code == 200:
            result = response.json()
            assistant_message = result["choices"][0]["message"]["content"]
            
            # Update session
            session["messages"].append({
                "role": "assistant",
                "content": assistant_message
            })
            
            logger.info("Follow-up response completed")
            return JSONResponse(status_code=200, content={
                "response": assistant_message,
                "status": "success"
            })
        else:
            logger.error(f"API error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=response.status_code, detail="Processing failed")
    
    except HTTPException as he:
        logger.error(f"HTTP Exception: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/clear_session")
async def clear_session(session_id: str = Form(...)):
    """Clear conversation history"""
    if session_id in conversation_sessions:
        del conversation_sessions[session_id]
    return {"status": "success", "message": "Session cleared"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "DocuMind AI - Document Intelligence Assistant",
        "version": "1.0.0"
    }

@app.get("/stats")
async def get_stats():
    """Get application statistics"""
    return {
        "active_sessions": len(conversation_sessions),
        "status": "operational"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

