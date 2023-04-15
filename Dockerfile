# Use an official Python runtime as the base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     cmake \
#     git \
#     libsm6 \
#     libxext6 \
#     libxrender-dev

# Download yolov3.cfg and coco.names files
RUN wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -O /app/yolov3.cfg && \
    wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names -O /app/coco.names && \
    wget https://pjreddie.com/media/files/yolov3.weights -O /app/yolov3.weights

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Copy your Python script for running YOLO
COPY . /app

# Set the entry point to your Python script
ENTRYPOINT ["python", "/app/app.py"]
