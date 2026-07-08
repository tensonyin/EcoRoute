# Use a lightweight official Python image
FROM python:3.12-slim-bookworm

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies, utilizing pre-built llama-cpp-python CPU wheels to avoid source compile overhead
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Copy download helper and download the 1.5B model weights during build phase
COPY download_model.py .
RUN python download_model.py

# Copy the rest of the application code
COPY src/ ./src/
COPY agent.py .

# Run the agent in unbuffered mode to ensure logs flush immediately
CMD ["python", "-u", "agent.py"]
