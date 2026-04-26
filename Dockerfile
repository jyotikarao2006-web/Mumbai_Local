# ── Mumbai Local OpenEnv — Hugging Face Spaces Dockerfile ────────────────────
# Deploys the Flask dashboard as a public HF Space (port 7860)

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
        matplotlib>=3.8.0

# --------------------------------------------------------------------------- #
# Application files                                                            #
# --------------------------------------------------------------------------- #
COPY app.py environment.py ./
COPY templates/ templates/
COPY static/ static/
COPY training_results.png .

# --------------------------------------------------------------------------- #
# Hugging Face Spaces requires the app to listen on port 7860                 #
# --------------------------------------------------------------------------- #
ENV PORT=7860
EXPOSE 7860

# --------------------------------------------------------------------------- #
# Entrypoint                                                                   #
# --------------------------------------------------------------------------- #
CMD ["python", "app.py", "--port", "7860"]
