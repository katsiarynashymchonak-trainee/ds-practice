# Use the official Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /ds-practice

RUN rm -rf /root/.cache

# Install system dependencies required for building Python packages and visualization
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy dependency file into the container
COPY requirements.txt .

RUN pip uninstall -y -r requirements.txt || true && \
    pip cache purge && \
    pip install --no-cache-dir -r requirements.txt

# Upgrade pip and essential packaging tools
RUN pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir -r requirements.txt \
    dvc[gs,gdrive] \
    apache-airflow==2.7.2 \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.7.2/constraints-3.11.txt"

# Check for broken dependencies (non-blocking)
RUN pip check || true

# Purge pip cache to reduce image size
RUN pip cache purge

# Copy the entire project into the container
COPY . .

# Optional: install your package if setup.py or pyproject.toml is present
# RUN pip install -e .

# Expose port 8000 (useful if running a web API)
EXPOSE 8000

# Default command: run the pipeline module
CMD ["python", "-m", "pipe.src.future_sales_inno_ds_project.pipeline"]
