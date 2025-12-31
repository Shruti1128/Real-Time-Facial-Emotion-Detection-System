Real-Time Facial Emotion Detection System
 Overview

This project is a real-time facial emotion detection system that uses a Convolutional Neural Network (CNN) to recognize human emotions from live webcam input.
It focuses on building a stable, realistic emotion recognition pipeline, addressing common challenges such as noisy predictions, dataset bias, and real-world deployment issues.

 Features

Real-time webcam-based emotion detection

Face detection using Haar Cascade / MTCNN

CNN-based emotion classification

Temporal smoothing for stable predictions

Emotion result locking with restart & quit controls

Emotion logging with timestamps

Modular and extensible code structure

 System Architecture
Webcam Input
     ↓
Face Detection (Haar / MTCNN)
     ↓
CNN Emotion Classifier
     ↓
Final Emotion Output

 Tech Stack

Language: Python

Computer Vision: OpenCV

Deep Learning: TensorFlow, Keras

Model: CNN

Face Detection: Haar Cascade / MTCNN

Data Processing: NumPy

Development Environment: VS Code, Virtualenv

 Emotions Supported

Happy

Sad

Angry

Neutral

Surprise

Disgust / Fear (dataset-dependent)

Note: Facial emotion recognition in real-world settings is inherently ambiguous.
This system prioritizes stable affective states over perfect fine-grained classification.

 Project Structure
emotion_model/
├── train.py
├── model.py
├── test_webcam.py
├── test_webcam_mtcnn.py
├── emotion_model.h5
├── dataset/
│   ├── train/
│   └── test/
├── results.csv
├── requirements.txt
└── README.md

 How to Run

Install dependencies

pip install -r requirements.txt


Train the emotion detection model

python train.py


Run real-time emotion detection

python test_webcam_mtcnn.py

 Known Limitations

Emotion recognition can be affected by lighting and camera quality

Subtle facial expressions may be misclassified

Dataset bias impacts rare emotions such as Fear and Disgust

These limitations are expected and documented.

 Use Cases

Emotion-aware applications

Human–computer interaction research

Behavioral analysis systems

Educational computer vision demos

 Learning Outcomes

Built an end-to-end real-time computer vision system

Gained experience handling dataset bias and noisy labels

Implemented temporal smoothing for stable predictions

Worked with real-world webcam inference challenges

 License

This project is intended for educational and research purposes.

 Author

Shruti Jha
