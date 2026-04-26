FROM python:3.11-slim
# --------------------------------------------------------------------------- #
# System deps                                                                  #
# --------------------------------------------------------------------------- #
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# --------------------------------------------------------------------------- #
# Working directory                                                            #
# --------------------------------------------------------------------------- #
WORKDIR /app
# --------------------------------------------------------------------------- #
# Python dependencies                                                          #
# Install lightweight deps first; skip torch/trl/unsloth for the Space        #
# (GPU training is done in Colab — the Space is for the live dashboard only)  #
# --------------------------------------------------------------------------- #
COPY requirements.txt .
RUN pip install --no-cache-dir \
        flask>=3.0.0 \
        numpy>=1.26.0 \
        matplotlib>=3.8.0 \
        streamlit>=1.28.0 \
        pandas>=2.0.0
# --------------------------------------------------------------------------- #
# Application files                                                            #
# --------------------------------------------------------------------------- #
COPY app.py environment.py gtfs_loader.py inference.py train.py ./
COPY training_log.json openenv.yaml pyproject.toml ./
COPY templates/ templates/
COPY static/ static/
# NOTE: training_results.png removed - it's a build artifact, not needed for deployment
# --------------------------------------------------------------------------- #
# Hugging Face Spaces requires the app to listen on port 7860                 #
# --------------------------------------------------------------------------- #
ENV PORT=7860
EXPOSE 7860
# --------------------------------------------------------------------------- #
# Entrypoint                                                                   #
# --------------------------------------------------------------------------- #
CMD ["python", "app.py", "--port", "7860"]
