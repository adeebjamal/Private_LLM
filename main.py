import logging
from dotenv import load_dotenv

# Load environment variables from .env file before anything else
load_dotenv()

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import StreamingResponse
from models.schemas import CreateConversationRequest, LoadMessagesRequest, AskQuestionRequest, RenameConversationRequest, DeleteConversationRequest
import database
import model
import time
import json

# Setup logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

# Lifespan context manager for startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up server...")
    try:
        # 1. Initialize database tables
        database.init_db()
    except Exception as e:
        logger.error(f"Failed to initialize database on startup. Ensure DATABASE_URL is set correctly: {e}")
        
    # 2. Load the LLM
    model.load_model()
    
    yield
    # Shutdown
    logger.info("Shutting down server...")

# Initialize FastAPI app
app = FastAPI(title = "AI Assistant API", lifespan = lifespan)

# Add CORS middleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],)



# --- Endpoints ---

@app.get("/conversations")
async def get_conversations():
    """Returns all conversations with their IDs, titles, message count."""
    try:
        conversations = database.get_all_conversations()
        # Format dates as ISO strings
        for c in conversations:
            if 'created_at' in c and c['created_at']:
                c['created_at'] = c['created_at'].isoformat()
        return {"conversations": conversations}
    except Exception as e:
        logger.error(f"Error fetching conversations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/conversations/create")
async def create_conversation(req: CreateConversationRequest):
    """Creates a new conversation topic and stores it in DB."""
    if not req.title or not req.title.strip():
        raise HTTPException(status_code=400, detail="title must not be empty")
        
    try:
        conv = database.create_conversation(req.title)
        if 'created_at' in conv and conv['created_at']:
            conv['created_at'] = conv['created_at'].isoformat()
        return conv
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/conversations/messages")
async def load_messages(req: LoadMessagesRequest):
    """Returns paginated messages from a specific conversation."""
    if req.start_row < 1:
        raise HTTPException(status_code=400, detail="start_row must be >= 1")
    if req.end_row < req.start_row:
        raise HTTPException(status_code=400, detail="end_row must be >= start_row")
        
    try:
        # Check if conversation exists
        conv = database.get_conversation(req.conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="conversation_id must exist")
            
        result = database.get_messages_paginated(req.conversation_id, req.start_row, req.end_row)
        
        # Format dates
        for msg in result["messages"]:
            if 'created_at' in msg and msg['created_at']:
                msg['created_at'] = msg['created_at'].isoformat()
                
        return {
            "conversation_id": req.conversation_id,
            "conversation_title": conv["title"],
            "start_row": req.start_row,
            "end_row": req.end_row,
            "total_messages": result["total_messages"],
            "messages": result["messages"]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading messages: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/conversations/ask")
async def ask_question(req: AskQuestionRequest):
    """Generates response via LLM and saves to DB."""
    logger.info(f"Received /ask request for conversation_id: {req.conversation_id}")
    start_req_time = time.time()
    
    if not req.query or not req.query.strip():
        logger.warning("Empty query received in /ask")
        raise HTTPException(status_code=400, detail="query must not be empty")
        
    try:
        # 1. Check if conversation exists
        conv = database.get_conversation(req.conversation_id)
        if not conv:
            logger.warning(f"Conversation {req.conversation_id} not found in DB")
            raise HTTPException(status_code=404, detail="conversation_id must exist")
            
        # 2. Fetch last 10 messages for context
        history = database.get_messages(req.conversation_id, limit=10)
        logger.info(f"Fetched {len(history)} previous messages for context")
        
        # 3. Stream generator to keep connection alive during slow CPU inference
        def generate():
            logger.info("Starting LLM generation...")
            gen_start_time = time.time()
            
            # Yield initial space to start response and prevent proxy timeout
            yield " "
            
            full_response = ""
            last_yield_time = time.time()
            first_token_time = None
            chunk_count = 0
            
            for text_chunk in model.generate_response_stream(history, req.query, req.max_tokens):
                if first_token_time is None:
                    first_token_time = time.time()
                    logger.info(f"Time to first token: {first_token_time - gen_start_time:.2f} seconds")
                
                full_response += text_chunk
                chunk_count += 1
                
                # Yield a space every 15 seconds to keep connection open
                current_time = time.time()
                if current_time - last_yield_time > 15:
                    logger.info(f"Yielding keep-alive space. Time elapsed: {current_time - gen_start_time:.2f}s, Chunks generated: {chunk_count}")
                    yield " "
                    last_yield_time = current_time
                    
            gen_end_time = time.time()
            logger.info(f"LLM generation complete! Total generation time: {gen_end_time - gen_start_time:.2f} seconds, Total chunks: {chunk_count}")
                    
            # 4. Save to database
            db_start = time.time()
            saved_msg = database.save_message(req.conversation_id, req.query, full_response.strip())
            logger.info(f"Saved generated response to DB in {time.time() - db_start:.2f} seconds")
            
            # Yield the final JSON
            final_dict = {
                "conversation_id": req.conversation_id,
                "user_query": saved_msg["user_query"],
                "response": saved_msg["response"]
            }
            logger.info(f"Total request processing time: {time.time() - start_req_time:.2f} seconds. Sending final JSON response.")
            yield json.dumps(final_dict)
            
        return StreamingResponse(generate(), media_type="application/json")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.put("/conversations/rename")
async def rename_conversation(req: RenameConversationRequest):
    """Renames an existing conversation."""
    if not req.new_name or not req.new_name.strip():
        raise HTTPException(status_code=400, detail="new_name must not be empty")

    try:
        conv = database.get_conversation(req.conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="conversation_id not found")

        updated = database.rename_conversation(req.conversation_id, req.new_name.strip())
        if updated and 'created_at' in updated and updated['created_at']:
            updated['created_at'] = updated['created_at'].isoformat()
        return updated
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error renaming conversation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.delete("/conversations/delete")
async def delete_conversation(req: DeleteConversationRequest):
    """Deletes a conversation and all its messages."""
    try:
        conv = database.get_conversation(req.conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="conversation_id not found")

        deleted = database.delete_conversation(req.conversation_id)
        return {"message": "Conversation deleted successfully", "deleted_id": deleted["id"]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
