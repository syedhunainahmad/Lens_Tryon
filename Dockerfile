# Multi-stage Dockerfile for the FastAPI + MediaPipe + TFLite app
# Base: slim Python with required system packages for OpenCV/aiortc/av
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       libglib2.0-0 \
       libgl1 \
       ffmpeg \
       pkg-config \
       libsrtp2-dev \        
       libopus-dev \         
       libvpx-dev \          
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only dependency files first for better Docker layer caching
COPY requirements.txt packages.txt ./

RUN pip install --upgrade pip setuptools wheel \
	&& pip install -r requirements.txt

# Copy application source
COPY . /app

# Expose the port the app listens on (main.py runs uvicorn on 8001)
EXPOSE 8001

# Default command to run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--loop", "asyncio"]

