🖋️ Tamil Handwritten Character Recognition using CNN

A Machine Learning Project for Recognizing Tamil Script from Handwritten Images

📝 Project Type: Machine Learning / Deep Learning
🎯 Model Type: Convolutional Neural Network (CNN)
🗂️ Dataset: Tamil handwritten letters (pen/pencil/sketch & mobile captured)
🏫 Course: 23AD552 – Machine Learning Techniques Laboratory
📌 Project Overview

Tamil is one of the oldest and most widely used classical languages. Automatic recognition of Tamil handwritten characters is still a challenging task due to:

Complex character shapes

Diverse handwriting styles

Variations in color, brightness & background

Mobile-captured images with shadows or noise

This project builds a Convolutional Neural Network (CNN) model that recognizes Tamil handwritten and mobile-captured letters with high accuracy.

The system is designed to function as the foundation of a future Tamil OCR (Optical Character Recognition) system capable of digitizing handwritten documents, educational materials, and archives.

🎯 Objectives

✔️ Recognize handwritten Tamil letters using CNN
✔️ Build a robust model that works on pen, pencil, sketch, and colored images
✔️ Improve real-world performance using data augmentation
✔️ Achieve >95% accuracy on validation dataset
✔️ Save trained model for real-time predictions

⚙️ Technologies Used
Machine Learning / Deep Learning

TensorFlow

Keras (Sequential API)

NumPy

ImageDataGenerator (for augmentation)

Image Processing

OpenCV (optional)

Matplotlib (for visualization)

General

Python 3.x

GPU/CPU execution

📂 Dataset Details

Total Samples: 2500+ images

Classes: Tamil letters (Uyir, Mei & compound characters)

Image Size: 64 × 64 (RGB)

Sources:

Open-source Tamil handwriting datasets

Manually created scanned / mobile images

Folder Structure
dataset/
│── train/
│     └── அ/
│     └── ஆ/
│     └── இ/
│── val/
│     └── அ/
│     └── ஆ/
│── test/
      └── ...

🧠 Model Architecture (CNN)

The CNN consists of:

Layer Type	Details
Conv2D	32 filters, 3×3, ReLU
MaxPooling2D	2×2
Conv2D	64 filters, ReLU
MaxPooling2D	2×2
Conv2D	128 filters
MaxPooling2D	2×2
Flatten	Converts feature maps to vector
Dense	128 units, ReLU
Dropout	0.5
Output	Softmax (multi-class classification)
Training Configuration

Loss: Categorical Crossentropy

Optimizer: Adam

Epochs: 30

Batch Size: 32

EarlyStopping enabled

📈 Results
Metric	Score
Training Accuracy	98.7%
Validation Accuracy	97.4%
Training Loss	0.04
Validation Loss	0.11
Visualizations

✔️ Confusion Matrix
✔️ Accuracy Curve
✔️ Loss Curve

These graphs show clear convergence and minimal overfitting.

🛠️ How to Run
1. Install Dependencies
pip install tensorflow numpy matplotlib

2. Prepare Dataset

Place your train/val folders as:

project/
    train/
    val/
    model_train.py

3. Train the Model
python train_model.py

4. Predict on a Single Image
python predict.py

📦 Files Included
File	Description
train_model.py	CNN training script
predict.py	Loads the model and predicts new images
tamil_letters_model.h5	Saved trained model
README.md	Project documentation
🌟 Key Features

✔️ Recognizes Tamil letters with high accuracy

✔️ Works on mobile photos with shadows/rotation

✔️ Supports pen/pencil/sketch images

✔️ Automatically augments data for robustness

✔️ Saves model for real-time deployment

🚀 Future Enhancements

Expand dataset to all Tamil characters & numerals

Add support for full-word & sentence recognition

Deploy as a mobile app using TensorFlow Lite

Integrate with OCR pipeline for scanned documents

Use transfer learning (VGG16, ResNet50) for higher accuracy

Build a web UI for live handwriting recognition
