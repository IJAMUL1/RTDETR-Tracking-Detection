# Object-Tracking-Detection-Transformers-RT-DETR-

RTDETR ingeniously adapts Vision Transformers (ViT), famous for their success in image classification, into the domain of real-time object detection. This marks a shift from traditional anchor-based object detectors (like YOLO).This project utilizes the RTDETR model for real-time object detection and tracking in videos. It involves utilizing a pre-trained rtdetr model trained on the coco dataset.  

![Untitled video - Made with Clipchamp (2)](https://github.com/IJAMUL1/RANSAC-Point-Cloud-Plane-Fitting/assets/60096099/3a881f53-ec95-4bf9-b40f-d66a0e579557)


## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project demonstrates real-time object detection in videos using the RTDETR model, a transformer-based object detector. It allows users to process videos, detect objects in real-time, and visualize bounding boxes with corresponding labels.

## Requirements
supervision, ultralytics, pytorch, opencv

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/RTDETR-Object-Detection.git
cd Object-Tracking-Detection-Transformers-RT-DETR-

## Usage
To use the RTDETR Object Detection:

Ensure that you have a CUDA-compatible GPU if you want to use CUDA for faster processing.
Run the Object_Tracking_RTDETR.py script, providing the path to the video file as an argument.

# Example usage:
video_path = r'path/to/your/video.mp4'
detector = Detection_Transformer(capture_index=0)
detector(video_path)








