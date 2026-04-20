# AI Assistant Backend

This is the FastAPI backend server for a personal AI assistant, designed to serve an Android app client. It uses a Supabase PostgreSQL database for storing conversation histories and leverages `llama-cpp-python` to run a highly optimized 4-bit GGUF version of **Llama-3.1-8B-Instruct** directly on the CPU.

## Prerequisites
- Python 3.10+
- Supabase Project (for PostgreSQL)

## Setup Instructions

1. **Clone the repository and navigate to it:**
   ```bash
   cd Private_LLM
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Environment Variables:**
   You must set the `DATABASE_URL` environment variable for the Supabase database connection.
   (The model `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` is loaded by default. You can override this using `MODEL_ID` and `MODEL_FILENAME` environment variables).
   
   ```bash
   # On Windows (PowerShell)
   $env:DATABASE_URL="postgresql://postgres.[YOUR_PROJECT_REF]:[YOUR_PASSWORD]@aws-0-eu-central-1.pooler.supabase.com:5432/postgres"
   
   # On Linux/macOS
   export DATABASE_URL="postgresql://..."
   ```

5. **Run the server locally:**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```
   *Note: On startup, the server automatically initializes the database tables and loads the LLM to memory. The first run will download the model weights.*

## Supabase Configuration
To get your `DATABASE_URL`:
1. Go to your Supabase project dashboard.
2. Click on **Project Settings** -> **Database**.
3. Under **Connection string**, select **URI**.
4. Replace `[YOUR-PASSWORD]` with your actual database password.

## Deploying to HuggingFace Spaces
1. Create a new Space on HuggingFace and select **Docker**.
2. Make sure you add a `Dockerfile` that runs the FastAPI app.
3. Upload all the files from this project to the Space.
4. In the Space settings, add **Repository Secrets**:
   - Name: `DATABASE_URL`
   - Value: Your Supabase connection string.
5. The Space will build and deploy automatically.

## API Endpoints

### 1. Health Check
```bash
curl -X GET http://localhost:8000/health
```

### 2. Get All Conversations
```bash
curl -X GET http://localhost:8000/conversations
```

### 3. Create Conversation
```bash
curl -X POST http://localhost:8000/conversations/create \
     -H "Content-Type: application/json" \
     -d '{"title": "Coding Questions"}'
```

### 4. Load Messages (Paginated)
```bash
curl -X POST http://localhost:8000/conversations/messages \
     -H "Content-Type: application/json" \
     -d '{"conversation_id": 1, "start_row": 1, "end_row": 10}'
```

### 5. Ask Question
```bash
curl -X POST http://localhost:8000/conversations/ask \
     -H "Content-Type: application/json" \
     -d '{"conversation_id": 1, "query": "What is Python?", "max_tokens": 500}'
```
