# Coffee Bean Classification Project

## Project Overview
This project focuses on classifying coffee beans into four categories: Dark, Green, Light, and Medium. Using a Convolutional Neural Network (CNN), we train a model to recognize and classify images of coffee beans.

## Dataset
The dataset consists of images divided into four classes, each within its respective folder inside the `coffe_model` directory. The structure is as follows:
```
coffe_model/
│
├── train/
│   ├── Dark/
│   ├── Green/
│   ├── Light/
│   └── Medium/
│
└── test/
    ├── Dark/
    ├── Green/
    ├── Light/
    └── Medium/
```
A CSV file, `Coffee Bean.csv`, contains the image paths and their corresponding labels.

## Prerequisites
- Python 3.6+
- TensorFlow 2.15.0
- NumPy
- Pandas
- scikit-learn

## Installation
To set up the project environment, run the following commands:
```bash
pip install tensorflow==2.15.0 numpy pandas scikit-learn
```

## Usage
To train the model and evaluate its performance, run:
```bash
python main.py
```

## Model
The CNN model is defined in `main.py`. It consists of convolutional layers followed by max-pooling layers, a flattening layer, and dense layers for classification.

## TensorFlow Lite Conversion
The trained model is converted to TensorFlow Lite format for deployment on mobile or embedded devices.
