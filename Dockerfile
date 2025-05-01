# Use a lightweight Python image
FROM python:3.12-slim

# Install system dependencies (MUST keep this!)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
# Note: PyTorch is installed from the CPU-only index to save space
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the app
COPY . .

# Run FastAPI on port 10000 (Render's default)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]