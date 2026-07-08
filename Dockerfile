# Use a lightweight official Python image
FROM python:3.12-slim-bookworm

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the agent code
COPY agent.py .

# Run the agent in unbuffered mode to ensure logs flush immediately
CMD ["python", "-u", "agent.py"]
