#!/bin/bash
# Ollama Model Preloader
# This script preloads specified models into Ollama on container startup
# Usage: docker-compose exec ollama /app/scripts/ollama-preload.sh

set -e

MODELS="${OLLAMA_MODELS:-llama2,mistral}"

echo "==================================="
echo "Ollama Model Preloader"
echo "==================================="
echo "Models to preload: $MODELS"
echo ""

# Wait for Ollama service to be ready
echo "Waiting for Ollama service..."
until curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; do
  echo "  Ollama not ready yet, waiting..."
  sleep 2
done
echo "✓ Ollama service is ready"
echo ""

# Parse comma-separated models and pull each one
IFS=',' read -ra MODEL_ARRAY <<<"$MODELS"
total=${#MODEL_ARRAY[@]}
current=0

for model in "${MODEL_ARRAY[@]}"; do
  current=$((current + 1))
  model=$(echo "$model" | xargs) # Trim whitespace

  echo "[$current/$total] Pulling model: $model"

  if ollama pull "$model"; then
    echo "  ✓ Successfully pulled: $model"
  else
    echo "  ✗ Failed to pull: $model"
    # Continue with other models even if one fails
  fi
  echo ""
done

echo "==================================="
echo "Preloading complete!"
echo "==================================="
echo ""
echo "Available models:"
ollama list
