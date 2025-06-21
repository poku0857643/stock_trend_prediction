# Dockerfile for frontend
FROM python:3.11-slim

WORKDIR /app


# Copy static files \
COPY . .


# Expose port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000')" || exit 1

# Start simple HTTP server
CMD ["python", "-m", "http.server", "8000"]