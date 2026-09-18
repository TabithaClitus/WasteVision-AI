                                                            ♻️ WasteVision AI

Real-Time Waste Instance Segmentation & Smart Waste Analytics

WasteVision AI is an AI-powered computer vision system designed to detect, classify, and segment individual waste objects in real-world and cluttered environments. The system uses YOLO11s Instance Segmentation to identify individual waste items, generate pixel-level masks, and provide waste analytics from images and videos.

🚀 Live Demo

https://wastevision-ai-pn7d4klffrqhb2rd3oh79g.streamlit.app/

🎯 Problem Statement

IS-02: Real-World Waste Instance Segmentation

Real-world waste is often found in cluttered environments where objects may overlap, partially occlude one another, or have irregular shapes. Traditional object detection using only bounding boxes cannot precisely identify the visible region of each waste item.

WasteVision AI addresses this problem using instance segmentation, where every detected waste object receives:

Individual segmentation mask

Waste category

Confidence score

Bounding box

Object count

Visual analytics

✨ Key Features

🔍 Waste Detection – Detects individual waste objects.

🎭 Instance Segmentation – Generates a separate pixel-level mask for each detected object.

🏷️ Waste Classification – Groups waste into 14 practical categories.

🎥 Video Analysis – Processes uploaded videos frame-by-frame.

📊 Waste Analytics – Displays detected object counts and category distribution.

⚙️ Adjustable Confidence Threshold – Allows users to control detection sensitivity.

📥 Processed Video Export – Generates a browser-compatible segmented MP4.

🌐 Web Interface – Interactive Streamlit dashboard for easy demonstration.

🧠 Model

WasteVision AI uses YOLO11s-seg, an Ultralytics YOLO model trained specifically on the prepared waste dataset.

Dataset

The project uses the TACO (Trash Annotations in Context) dataset.

Dataset source:

https://www.kaggle.com/datasets/sohamchaudhari2004/taco-trash-detection-dataset

Dataset Preparation

The original dataset contains 60 waste categories. These were consolidated into 14 broader categories to make the system more practical for real-world waste analytics.

Final Classes

ID	Class

0	Plastic Bag/Wrapper

1	Cigarette

2	Other/Unlabeled Waste

3	Other Plastic

4	Plastic Bottle

5	Plastic Cap/Lid

6	Other Metal

7	Metal Can

8	Glass

9	Cardboard/Carton

10	Plastic Straw/Utensil

11	Paper

12	Styrofoam

13	Plastic Cup

📈 Model Performance

Evaluation was performed on a held-out 150-image test set containing 521 annotated waste instances.

Metric	Result

Mask mAP@50	29.88%

Mask mAP@50–95	21.09%

Box mAP@50	32.21%

Box mAP@50–95	25.18%

What these metrics mean

Mask mAP@50 measures segmentation performance when the predicted mask has at least 50% IoU with the ground-truth mask.

Mask mAP@50–95 evaluates mask precision across stricter IoU thresholds from 50% to 95%.

Box mAP@50 evaluates bounding-box detection at an IoU threshold of 50%.

Box mAP@50–95 evaluates bounding-box detection across IoU thresholds from 50% to 95%.

SYSTEM ARCHITECTURE

<img width="528" height="482" alt="image" src="https://github.com/user-attachments/assets/77b70d91-3c1f-42d0-9c7f-259ec51c6afa" />

🛠️ Tech Stack

Python

PyTorch

Ultralytics YOLO

YOLO11s Instance Segmentation

OpenCV

NumPy

Streamlit

FFmpeg / imageio-ffmpeg

Google Colab for model training

📂 Project Structure

WasteVision-AI/

│

├── app.py

├── best.pt

├── requirements.txt

└── README.md

File Description

File	Purpose

app.py	Streamlit application

best.pt	Trained YOLO11s segmentation model

requirements.txt	Python dependencies

README.md	Project documentation

⚙️ Installation

Clone the repository:

git clone https://github.com/TabithaClitus/WasteVision-AI.git
cd WasteVision-AI

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

The application will open in your browser.

📦 Requirements

streamlit

ultralytics

opencv-python-headless

numpy

imageio-ffmpeg

🔄 Workflow

Image Analysis

Upload Image

     ↓
     
YOLO11s-seg

     ↓
     
Detect Waste Objects

     ↓
     
Generate Individual Masks

     ↓
     
Classify Objects

     ↓
     
Display Results

     ↓
     
Generate Analytics

Video Analysis

Upload Video

     ↓
     
Read Video Frames

     ↓
     
YOLO11s-seg Detection

     ↓
     
Generate Segmentation Masks

     ↓
     
Annotate Each Frame

     ↓
     
Encode as H.264 MP4

     ↓
     
Display Segmented Video

     ↓
     
Generate Waste Analytics

📊 Analytics

The application provides category-level analytics such as:

Total detected waste instances

Individual waste categories

Number of detections per category

Confidence scores

Category distribution

For video processing, the displayed counts represent detection events across processed frames, rather than unique physical objects tracked throughout the entire video.


👩‍💻 Project

WasteVision AI

Real-Time Waste Instance Segmentation & Smart Waste Analytics

Built using YOLO11s Instance Segmentation + Streamlit.
