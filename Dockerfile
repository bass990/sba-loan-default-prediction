# Lean inference image for the SBA loan-default API.
# Build:  docker build -t sba-api .
# Run:    docker run --rm -p 8000:8000 sba-api
# Health: curl http://localhost:8000/health

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install runtime deps first so a code edit doesn't bust the package layer.
COPY requirements-api.txt ./
RUN pip install -r requirements-api.txt

# Copy only what the API needs at runtime — not the notebook, scrapers, or data.
COPY score_model.py ./
COPY artifacts/ ./artifacts/
COPY api/ ./api/
# the Gradio scoring UI, served by the API at /ui (same image as the Hugging Face Space)
COPY huggingface-spaces/app.py huggingface-spaces/sample_500.csv ./huggingface-spaces/

EXPOSE 8000

# One Uvicorn worker is correct for a CPU-bound XGBoost service; scale
# horizontally (more containers) instead of more workers per container.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
