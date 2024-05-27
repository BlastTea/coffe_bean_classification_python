import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import os

# Load dataset
data = pd.read_csv('coffe_model/Coffee Bean.csv')

# Function to load and preprocess images
def preprocess_image(file_path):
    image = load_img(file_path, target_size=(224, 224))
    image = img_to_array(image)
    image /= 255.0  # Normalize to [0,1]
    return image

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='outputs/coffe_bean_detector.tflite')
interpreter.allocate_tensors()

# Get model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to run inference using TensorFlow Lite
def run_inference(image_data):
    interpreter.set_tensor(input_details[0]['index'], image_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

X = data['filepaths'].apply(lambda x: os.path.join('coffe_model', x)).values
y = data['class index'].values

# Load and preprocess images
X = np.array([preprocess_image(file_path) for file_path in X])
y = to_categorical(y, num_classes=4)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate the model on test data
correct_predictions = 0
for i in range(len(X_test)):
    # Get the preprocessed test image and expand dimensions
    test_image = np.expand_dims(X_test[i], axis=0).astype(np.float32)
    
    # Run inference
    prediction = run_inference(test_image)
    
    # Determine the predicted class
    predicted_class = np.argmax(prediction)
    true_class = np.argmax(y_test[i])
    
    # Increment correct predictions counter if prediction is right
    if predicted_class == true_class:
        correct_predictions += 1

# Calculate accuracy
accuracy = correct_predictions / len(X_test)
print(f"Test accuracy: {accuracy}")