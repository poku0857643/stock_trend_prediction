# Dockerfile for Trend Prediction and Strategy generator

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
#    PIP_NO_CACHE_DIR=1 \
#    PIP_DISABLE_PIP_VERSION_CHECK=1


# Set working directory
WORKDIR /app

## Install system dependencies
#RUN apt-get update && apt-get install -y --no-install-recommends \
#    gcc \
#    && rm -rf /var/lib/apt/lists/*


COPY apis/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir uvicorn[standard] \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

## Change ownership to non-root user
#RUN chown -R appuser:appuser /app
#USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
CMD curl -f http:localhost:8000/health ||exit 1

# Start the application
CMD ["uvicorn", "apis.main:app", "--host", "0.0.0.0", "--port", "8000"]