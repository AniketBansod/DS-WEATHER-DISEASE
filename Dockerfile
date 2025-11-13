FROM python:3.11-slim

# Prevents Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# System deps (build tools for scientific wheels if needed)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only minimal requirements for the runtime image
COPY requirements-prod.txt ./

RUN pip install --upgrade pip \
    && pip install -r requirements-prod.txt

# Copy application code and model artifacts
COPY app.py ./
COPY models ./models

# Expose the port Render provides via $PORT (default 8501 locally)
EXPOSE 8501

# Run Streamlit binding to 0.0.0.0 and the provided port
CMD ["sh", "-c", "streamlit run app.py --server.address 0.0.0.0 --server.port ${PORT:-8501}"]
