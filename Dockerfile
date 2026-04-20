# Use an official lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (Compilers for llama-cpp-python and Postgres drivers)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libpq-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Copy the rest of the application code
COPY . .

# HuggingFace requires applications to have write access to their own directories
# and runs them on port 7860 by default
RUN mkdir -p /app/model_cache && chmod 777 /app/model_cache

# Expose the default port used by HuggingFace Spaces
EXPOSE 7860

# Start the FastAPI server using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
