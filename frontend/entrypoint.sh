#!/bin/sh

# --- entrypoint.sh ---
# This script is the entrypoint for the Docker container.

# Change to the application directory
cd /app

# Execute the Uvicorn server.
# 'exec' replaces the shell process with the Uvicorn process,
# which is good practice for signal handling in containers.
echo "Starting Uvicorn server for NGSL Translator..."
exec uvicorn app:app --host 0.0.0.0 --port 8000
