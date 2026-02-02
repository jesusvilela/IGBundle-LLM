# Use NVIDIA CUDA base image with Python 3.11
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Symlink python3 to python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Install python dependencies
# Copy requirements first for cache
COPY requirements.txt .
# Install basic requirements
RUN pip install -r requirements.txt

# Install llama-cpp-python with CUDA support
# Force build with CUBLAS (now CUDA)
ENV CMAKE_ARGS="-DGGML_CUDA=on"
RUN pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

# Copy application code
COPY . .

# Expose ports
EXPOSE 7860
EXPOSE 8000

# Command to run the app
CMD ["python", "app.py"]
