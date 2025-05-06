# --- Dockerfile ---

# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    HF_HOME=/app/.cache/huggingface \
    PYTHONIOENCODING=UTF-8 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# Set the working directory
WORKDIR /app

# Install system dependencies (git for HF)
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies (python-dotenv is removed from requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Download the spaCy Japanese model
RUN python -m spacy download ja_core_news_sm

# Copy the config.ini file and its template
# Make sure config.ini is in your .dockerignore if it contains secrets!
COPY config.ini.template /app/config.ini.template
COPY config.ini /app/config.ini

# Copy the application code and static assets
COPY app.py .
COPY translator.py .
COPY frontend/ ./frontend/
COPY vocablists/ ./vocablists/
COPY entrypoint.sh .

# Make the entrypoint script executable
RUN chmod +x ./entrypoint.sh

# Create cache directory and set permissions (optional)
RUN mkdir -p /app/.cache/huggingface && chown -R nobody:nogroup /app/.cache
# USER nobody # Consider running as non-root user

# Expose the port the app runs on
EXPOSE 8000

# Set the entrypoint script
ENTRYPOINT ["./entrypoint.sh"]
