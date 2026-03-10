# ── Stage 1: Build frontend ──
FROM node:20-slim AS frontend-build
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# ── Stage 2: Python runtime ──
FROM python:3.12-slim
WORKDIR /app

# Install ffmpeg (required for video processing)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend
COPY backend/ ./backend/

# Copy built frontend into the location backend expects
COPY --from=frontend-build /build/dist ./frontend/dist/

# Create data directory (Railway mounts persistent volume at /data)
RUN mkdir -p /data/uploads /data/frames /data/models /data/upload_chunks

EXPOSE 8000

CMD cd /app/backend && uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
