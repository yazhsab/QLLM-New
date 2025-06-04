# Quantum LLM Dockerfile
# Enterprise-grade containerization for Quantum LLM

# Use Python 3.8 as base image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -r quantum-user && \
    chown -R quantum-user:quantum-user /app

# Switch to non-root user
USER quantum-user

# Expose API port
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["python", "-m", "quantum_llm.deployment.docker.entrypoint"]

# Set default command
CMD ["api"]