FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set Python3 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Install PyTorch and diffusers
RUN pip install --upgrade pip && \
    pip install torch torchvision diffusers[torch] accelerate

# Set up workspace
WORKDIR /app
COPY ./*.py /app

# Default command
CMD ["python", "generate_image.py"]
