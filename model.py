import os
import logging
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

logger = logging.getLogger(__name__)

llm = None

def load_model():
    """Download (if needed) and load the GGUF model into memory."""
    global llm
    
    # We will use Llama-3.1-8B-Instruct in 4-bit GGUF by default if not set
    # The user can still set a different GGUF model via env variables if they want
    repo_id = os.environ.get("MODEL_ID", "bartowski/Llama-3.2-3B-Instruct-GGUF")
    filename = os.environ.get("MODEL_FILENAME", "Llama-3.2-3B-Instruct-Q4_K_M.gguf")
    cache_dir = "./model_cache"
    
    logger.info(f"Checking for model {repo_id} ({filename}) in {cache_dir}...")
    try:
        # Download the model from HuggingFace Hub (this uses the cache automatically)
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir
        )
        
        logger.info(f"Loading model into memory from {model_path}...")
        # Load the model via llama_cpp
        # n_ctx is the context window. Llama 3 handles 8k easily, we'll set 4096 for RAM safety
        # HARDCODE n_threads to 2. HF Spaces free tier only gives 2 vCPUs.
        # os.cpu_count() returns the host machine's cores (often 64+) which causes extreme thread thrashing and destroys performance.
        llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=2, 
            verbose=False
        )
        logger.info(f"Successfully loaded {filename}")
    except Exception as e:
        logger.error(f"Error loading GGUF model: {e}")
        llm = None

def generate_response_stream(history: list, query: str, max_new_tokens: int = 500):
    """
    Generate a response using Llama's native chat completion, yielding chunks.
    history format: [{"role": "user", "content": "msg"}, {"role": "assistant", "content": "msg"}]
    """
    global llm
    
    if not llm:
        logger.warning("Generate response called but model is not loaded. Returning placeholder.")
        yield "I am a placeholder AI assistant. Please ensure the model downloaded correctly."
        return
        
    # Append the new query to the history
    messages = history.copy()
    
    # Prepend system prompt if history is empty (optional but recommended for Llama 3)
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": "You are a helpful AI assistant."})
        
    messages.append({"role": "user", "content": query})
    
    try:
        # Use llama_cpp's built-in chat formatting with streaming enabled
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.7,
            stream=True
        )
        
        for chunk in response:
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta:
                yield delta["content"]
                
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        yield f"Error generating response: {e}"
