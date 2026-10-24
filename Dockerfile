FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy local empyrical library and install it in editable mode
COPY external/ /app/external/
RUN pip install -e /app/external/empyrical

# Copy the rest of the application code
COPY . .

# Set PYTHONPATH so absolute package imports resolve correctly inside the container
ENV PYTHONPATH=/app

EXPOSE 8000

# Start FastAPI via Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]