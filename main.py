import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file before anything else
load_dotenv()

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models.schemas import CreateConversationRequest, LoadMessagesRequest, AskQuestionRequest, RenameConversationRequest, DeleteConversationRequest
import database
import model
import time
import json
import uuid
import threading
from typing import List, Dict

try:
    from ddgs import DDGS
except Exception:  # pragma: no cover - graceful fallback when dependency missing
    DDGS = None

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

# --- In-memory task store for async ask processing ---
# Format: { task_id: { "status": "processing" | "completed" | "failed", "result": {...} | None, "error": str | None } }
task_store: dict[str, dict] = {}

WEB_SEARCH_MODE = os.environ.get("WEB_SEARCH_MODE", "auto").strip().lower()
WEB_SEARCH_MAX_RESULTS = int(os.environ.get("WEB_SEARCH_MAX_RESULTS", "5"))
WEB_SEARCH_GATE_MAX_TOKENS = int(os.environ.get("WEB_SEARCH_GATE_MAX_TOKENS", "3"))


def _should_use_web_search(query: str, history: list) -> bool:
    """Legacy gate kept for compatibility; /ask now uses client-provided use_internet."""
    logger.info(f"Deciding if we should use web search for query: {query}")
    if WEB_SEARCH_MODE == "always":
        return True
    if WEB_SEARCH_MODE == "off":
        return False

    gate_prompt = (
        "You are a routing classifier.\n"
        "Decide if internet/web search is required to answer the user's query well.\n"
        "Reply with exactly one word: YES or NO.\n\n"
        f"User query: {query}"
    )
    gate_output = ""
    try:
        for chunk in model.generate_response_stream(history, gate_prompt, WEB_SEARCH_GATE_MAX_TOKENS):
            gate_output += chunk
            if len(gate_output) >= 8:
                break
    except Exception as e:
        logger.warning(f"LLM gate failed, defaulting to no web search: {e}")
        return False

    verdict = gate_output.strip().upper()
    if verdict.startswith("YES"):
        logger.info("Web search required for query")
        return True
    if verdict.startswith("NO"):
        logger.info("Web search not required for query")
        return False

    logger.info(f"Unclear gate response '{gate_output}'. Defaulting to no web search.")
    return False


def _duckduckgo_search(query: str, max_results: int) -> List[Dict[str, str]]:
    logger.info(f"Fetching web search snippets from DuckDuckGo for query: {query}")
    if DDGS is None:
        logger.warning("ddgs is not installed. Skipping web search.")
        return []

    try:
        with DDGS() as ddgs:
            raw_results = ddgs.text(query, max_results=max_results) or []
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")
        return []

    normalized_results: List[Dict[str, str]] = []
    for item in raw_results:
        normalized_results.append(
            {
                "title": (item.get("title") or "").strip(),
                "url": (item.get("href") or "").strip(),
                "snippet": (item.get("body") or "").strip(),
            }
        )
    
    logger.info(f"DuckDuckGo search results: {normalized_results}")
    return normalized_results


def _build_query_with_web_context(user_query: str, web_results: List[Dict[str, str]]) -> str:
    logger.info(f"Building query with web context for user query: {user_query}")
    if not web_results:
        return user_query

    lines = [
        "Use the web snippets below only as supporting context.",
        "If snippets are insufficient, say so clearly.",
        "",
        "Web results:",
    ]
    for idx, result in enumerate(web_results, start=1):
        lines.append(f"{idx}. {result['title']}")
        lines.append(f"   URL: {result['url']}")
        lines.append(f"   Snippet: {result['snippet']}")
    lines.append("")
    lines.append(f"User question: {user_query}")
    return "\n".join(lines)


def _format_sources_block(web_results: List[Dict[str, str]]) -> str:
    logger.info(f"Rendering source URLs for transparency in final output")
    if not web_results:
        return ""

    source_lines = ["", "", "Sources:"]
    for idx, result in enumerate(web_results, start=1):
        if result["url"]:
            source_lines.append(f"{idx}. {result['title']} - {result['url']}")
    return "\n".join(source_lines)

def _process_ask_in_background(task_id: str, conversation_id: int, query: str, max_tokens: int, history: list, use_internet: bool):
    logger.info(f"[Task {task_id}] Background processing started for conversation {conversation_id}")
    start_time = time.time()
    
    try:
        # 0. Optional internet retrieval controlled by client request payload
        web_results: List[Dict[str, str]] = []
        query_for_model = query
        if use_internet:
            logger.info(f"[Task {task_id}] use_internet=true, running DuckDuckGo search")
            web_results = _duckduckgo_search(query, WEB_SEARCH_MAX_RESULTS)
            query_for_model = _build_query_with_web_context(query, web_results)
        else:
            logger.info(f"[Task {task_id}] use_internet=false, skipping web search")

        # 1. Generate full response from LLM (streaming internally, collecting all chunks)
        logger.info(f"[Task {task_id}] Starting LLM generation...")
        gen_start = time.time()
        full_response = ""
        chunk_count = 0
        first_token_time = None
        
        for text_chunk in model.generate_response_stream(history, query_for_model, max_tokens):
            if first_token_time is None:
                first_token_time = time.time()
                logger.info(f"[Task {task_id}] Time to first token: {first_token_time - gen_start:.2f}s")
            full_response += text_chunk
            chunk_count += 1
        
        logger.info(f"[Task {task_id}] LLM generation complete in {time.time() - gen_start:.2f}s, chunks: {chunk_count}")
        
        # 2. Save to database
        db_start = time.time()
        final_response = f"{full_response.strip()}{_format_sources_block(web_results)}".strip()
        saved_msg = database.save_message(conversation_id, query, final_response)
        logger.info(f"[Task {task_id}] Saved to DB in {time.time() - db_start:.2f}s")
        
        # 3. Update task store with completed result
        task_store[task_id] = {
            "status": "completed",
            "result": {
                "conversation_id": conversation_id,
                "user_query": saved_msg["user_query"],
                "response": saved_msg["response"]
            },
            "error": None
        }
        logger.info(f"[Task {task_id}] Total background processing time: {time.time() - start_time:.2f}s")
        
    except Exception as e:
        logger.error(f"[Task {task_id}] Background processing failed: {e}")
        task_store[task_id] = {
            "status": "failed",
            "result": None,
            "error": str(e)
        }



# --- Endpoints ---

@app.get("/")
async def root():
    return {"message": "Hello, World!"}

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

@app.post("/conversations/ask", status_code=202)
async def ask_question(req: AskQuestionRequest):
    """Accepts user query, delegates LLM generation to a background thread, and returns immediately with a task_id."""
    logger.info(f"Received /ask request for conversation_id: {req.conversation_id}")
    
    if not req.query or not req.query.strip():
        logger.warning("Empty query received in /ask")
        raise HTTPException(status_code=400, detail="query must not be empty")
        
    try:
        # 1. Check if conversation exists
        conv = database.get_conversation(req.conversation_id)
        if not conv:
            logger.warning(f"Conversation {req.conversation_id} not found in DB")
            raise HTTPException(status_code=404, detail="conversation_id must exist")
            
        # 2. Fetch last 10 messages for context (lightweight DB read, done before dispatching)
        history = database.get_messages(req.conversation_id, limit=10)
        logger.info(f"Fetched {len(history)} previous messages for context")
        
        # 3. Create a task ID and register it as "processing"
        task_id = str(uuid.uuid4())
        task_store[task_id] = {
            "status": "processing",
            "result": None,
            "error": None
        }
        
        # 4. Dispatch background thread for LLM generation + DB save
        thread = threading.Thread(
            target=_process_ask_in_background,
            args=(task_id, req.conversation_id, req.query, req.max_tokens, history, req.use_internet),
            daemon=True
        )
        thread.start()
        logger.info(f"Dispatched background task {task_id} for conversation {req.conversation_id}")
        
        # 5. Return immediately with 202 Accepted
        return {
            "status": "accepted",
            "task_id": task_id,
            "message": "Your question is being processed. Poll /conversations/ask/status/{task_id} for the result."
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/conversations/ask/status/{task_id}")
async def get_ask_status(task_id: str):
    """Poll this endpoint to check if the background LLM task has completed."""
    task = task_store.get(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="task_id not found")
    
    if task["status"] == "processing":
        return {
            "task_id": task_id,
            "status": "processing",
            "message": "Still generating response. Please poll again."
        }
    elif task["status"] == "completed":
        # Clean up from store after delivering result
        result = task["result"]
        del task_store[task_id]
        return {
            "task_id": task_id,
            "status": "completed",
            "result": result
        }
    else:  # failed
        error = task["error"]
        del task_store[task_id]
        raise HTTPException(status_code=500, detail=f"Task failed: {error}")

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
