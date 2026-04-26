FROM python:3.13-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY zebraid/ ./zebraid/

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Set environment variables
ENV KMP_DUPLICATE_LIB_OK=TRUE
ENV PYTHONUNBUFFERED=1

# Create persistent data directory
RUN mkdir -p /data/registry

# Expose port
EXPOSE 8000

# Start the server
CMD ["uvicorn", "zebraid.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
