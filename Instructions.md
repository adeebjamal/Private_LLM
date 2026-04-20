# 🤖 Claude Code Prompt — Build a ChatGPT Alternative
DB Password - iPhone16Pixel6A
-----

## PROMPT (Copy everything below this line)

-----

Build a complete backend server for a personal AI assistant app. This is a ChatGPT alternative that will be hosted on **HuggingFace Spaces** and connected to a **Supabase PostgreSQL database**. An Android app will consume the APIs via HTTP requests.

-----

## 🏗️ Tech Stack

- **Framework:** FastAPI (Python)
- **LLM:** Will decide later - loaded via transformers library
- **Database:** Supabase (PostgreSQL via psycopg2)
- **Hosting:** HuggingFace Spaces
- **Language:** Python 3.10+

-----

## 📁 Project Structure

Create the following files:

```
project/
├── main.py            — FastAPI server + all endpoints
├── model.py           — LLM loading and response generation
├── database.py        — All database operations
├── requirements.txt   — Dependencies
└── README.md          — Setup instructions
```

-----

## 🗄️ Database Schema

Create these two tables in Supabase. Include the CREATE TABLE statements in database.py inside the `init_db()` function:

```sql
-- Table 1: Conversations
-- Stores different chat topics (Coding, Gaming, Miscellaneous etc)
CREATE TABLE IF NOT EXISTS conversations (
    id         SERIAL PRIMARY KEY,
    title      VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Table 2: Messages
-- Each row = one complete Q&A exchange
-- Belongs to a conversation via foreign key
CREATE TABLE IF NOT EXISTS messages (
    id              SERIAL PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    user_query      TEXT NOT NULL,
    response        TEXT NOT NULL,
    created_at      TIMESTAMP DEFAULT NOW()
);
```

**Relationships:**

- One conversation has many messages (one-to-many)
- Deleting a conversation deletes all its messages (CASCADE)

-----

## 📡 API Endpoints

Implement exactly these 5 endpoints:

-----

### Endpoint 1 — Get Conversation List

```
GET /conversations
```

**Description:** Returns all conversations with their IDs, titles, message count.

**No request body.**

**Response:**

```json
{
    "conversations": [
        {
            "id"            : 1,
            "title"         : "Coding",
            "message_count" : 10,
            "created_at"    : "2026-01-01T10:00:00"
        },
        {
            "id"            : 2,
            "title"         : "Gaming",
            "message_count" : 5,
            "created_at"    : "2026-01-01T10:00:00"
        }
    ]
}
```

-----

### Endpoint 2 — Create Conversation

```
POST /conversations/create
```

**Description:** Creates a new conversation topic and stores it in DB.

**Request body:**

```json
{
    "title": "Coding"
}
```

**Response:**

```json
{
    "id"        : 1,
    "title"     : "Coding",
    "created_at": "2026-01-01T10:00:00"
}
```

**Validation:**

- title must not be empty — return 400 if empty

-----

### Endpoint 3 — Load Messages (Paginated)

```
POST /conversations/messages
```

**Description:** Returns paginated messages from a specific conversation.
Used by the Android app to load chat history.

**Request body:**

```json
{
    "conversation_id" : 1,
    "start_row"       : 1,
    "end_row"         : 10
}
```

**Response:**

```json
{
    "conversation_id"   : 1,
    "conversation_title": "Coding",
    "start_row"         : 1,
    "end_row"           : 10,
    "total_messages"    : 25,
    "messages": [
        {
            "id"        : 1,
            "user_query": "What is multithreading?",
            "response"  : "Multithreading is...",
            "created_at": "2026-01-01T10:00:00"
        }
    ]
}
```

**Validation:**

- conversation_id must exist — return 404 if not found
- start_row must be >= 1 — return 400 if not
- end_row must be >= start_row — return 400 if not

**Pagination logic:**

- Use SQL OFFSET and LIMIT
- OFFSET = start_row - 1 (convert from 1-based to 0-based)
- LIMIT  = end_row - start_row + 1

-----

### Endpoint 4 — Ask Question

```
POST /conversations/ask
```

**Description:**

1. Receives a user query and conversation_id
1. Fetches last 10 messages from that conversation as context
1. Builds a prompt with full conversation history
1. Sends to LLM and generates response
1. Saves user_query + response as one row in messages table
1. Returns the response

**Request body:**

```json
{
    "conversation_id" : 1,
    "query"           : "What is multithreading in Java?",
    "max_tokens"      : 500
}
```

**Response:**

```json
{
    "conversation_id" : 1,
    "user_query"      : "What is multithreading in Java?",
    "response"        : "Multithreading in Java is..."
}
```

**Validation:**

- conversation_id must exist — return 404 if not found
- query must not be empty — return 400 if empty
- max_tokens defaults to 500 if not provided

-----

### Endpoint 5 — Health Check

```
GET /health
```

**Response:**

```json
{
    "status": "healthy"
}
```

-----

## 🧠 LLM Implementation (model.py)

### Model Loading

- Load `microsoft/phi-2` from HuggingFace using `AutoModelForCausalLM` and `AutoTokenizer`
- Use `device_map="cpu"` — no GPU available on free tier
- Use `torch_dtype=torch.float32` for CPU compatibility
- Set `trust_remote_code=True`
- Load model once on server startup — not on every request

### Prompt Building

Build prompts that include full conversation history so the model remembers previous messages:

```
You are a helpful AI assistant.

User: <previous message 1>
Assistant: <previous response 1>
User: <previous message 2>
Assistant: <previous response 2>
User: <new question>
Assistant:
```

### Generation Parameters

- `max_length`: 2048 (input context window)
- `max_new_tokens`: controlled by user via request (default 500)
- `temperature`: 0.7
- `do_sample`: True
- `pad_token_id`: tokenizer.eos_token_id

### Response Decoding

- Decode only newly generated tokens — not the input prompt
- Strip special tokens
- Strip whitespace

-----

## 🗄️ Database Implementation (database.py)

### Connection

- Read `DATABASE_URL` from environment variable
- Use `psycopg2` for PostgreSQL connection
- Use `psycopg2.extras.RealDictCursor` so rows return as dicts

### Functions to implement:

```python
init_db()                          — Create tables if not exist
create_conversation(title)         — Insert new conversation, return it
get_all_conversations()            — Return all conversations with message count
get_conversation(id)               — Return single conversation by ID
save_message(conv_id, query, resp) — Insert one Q&A row into messages
get_messages(conv_id, limit)       — Return last N messages as LLM history format
get_messages_paginated(conv_id, start_row, end_row) — Return paginated messages
```

### Important — Two Different Message Formats:

**Format 1 — For LLM context** (used in /ask endpoint):

```python
# Returns alternating user/assistant format for prompt building
[
    {"role": "user",      "content": "What is Java?"},
    {"role": "assistant", "content": "Java is..."},
    {"role": "user",      "content": "Give an example"},
    {"role": "assistant", "content": "Sure..."}
]
```

**Format 2 — For API response** (used in /messages endpoint):

```python
# Returns raw rows for display in Android app
[
    {
        "id"        : 1,
        "user_query": "What is Java?",
        "response"  : "Java is...",
        "created_at": "..."
    }
]
```

-----

## ⚙️ Server Setup (main.py)

### Startup Sequence

On server startup (using lifespan):

1. Call `init_db()` — create tables if not exist
1. Call `load_model()` — load LLM into memory

### CORS

Enable CORS for all origins — the Android app needs to call this API:

```python
allow_origins=["*"]
allow_methods=["*"]
allow_headers=["*"]
```

### Pydantic Models

Define request/response models for all endpoints using Pydantic BaseModel.

-----

## 📦 requirements.txt

```
fastapi==0.110.0
uvicorn==0.29.0
transformers==4.40.0
torch==2.2.2
accelerate==0.29.0
pydantic==2.6.4
huggingface-hub==0.22.2
psycopg2-binary==2.9.9
```

-----

## 🔐 Environment Variables

The server reads these from environment variables:

- `DATABASE_URL` — Supabase PostgreSQL connection string

Format:

```
postgresql://user:password@host:5432/dbname
```

On HuggingFace Spaces — add this as a **Repository Secret** in Space settings.

-----

## 📝 README.md

Generate a README with:

- Project overview
- How to set up Supabase and get DATABASE_URL
- How to deploy to HuggingFace Spaces
- How to add DATABASE_URL as a secret on HuggingFace
- All API endpoints with example curl commands

-----

## ✅ Additional Requirements

1. **Error handling** — wrap all DB operations in try/except and return meaningful error messages
1. **Input validation** — validate all request fields before processing
1. **HTTP status codes** — use correct codes (200, 201, 400, 404, 500)
1. **No hardcoded values** — all config via environment variables
1. **Clean code** — add comments explaining what each function does
1. **Single file model loading** — model loads once on startup, reused for all requests

-----

## 🚀 Final Note

This server will be consumed by an Android app built with Kotlin and Jetpack Compose using Retrofit for HTTP calls. Make sure all responses are clean JSON so Retrofit can deserialize them easily.