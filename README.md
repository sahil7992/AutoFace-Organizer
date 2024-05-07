
# AutoFace Organizer

## Overview
AutoFace Organizer is a Python-based facial recognition tool designed to automate the learning and detection of faces. It uses advanced machine learning models to capture and recognize facial features from images stored in a specified directory.

## Features
- **Face Learning:** Analyzes images from a specified folder, extracts face embeddings using deep learning models, and saves the embeddings for later recognition.
- **Face Detection:** Recognizes faces in new images by comparing with previously learned embeddings and identifies the person with highlighted bounding boxes.

## Installation

### Prerequisites
- Python 3.6 or higher
- PyTorch
- facenet_pytorch
- OpenCV
- Numpy
- sklearn

### Setup
To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/sahil7992/AutoFace-Organizer
   ```


## Usage

### Learning Faces
Run the learning script to process images and learn face embeddings:
```bash
python learn.py
```
Make sure your images are stored in the specified `face_folder_path` inside `learn.py`, with each image named according to the person's name.

### Detecting Faces
Execute the detection script to identify and recognize faces in new images:
```bash
python detect.py
```
Update the `folder_path` in `detect.py` to the directory containing images you want to process for detection.

## Configuration
Adjust the following configurations in the scripts as per your needs:
- `face_folder_path` in `learn.py`: Path to the directory containing the labeled images for learning.
- `folder_path` in `detect.py`: Path to the directory for processing detection images.


