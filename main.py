import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical, load_img, img_to_array
import pandas as pd
import numpy as np
import os

# Load dataset
data = pd.read_csv('coffe_model/Coffee Bean.csv')

# Preprocess function to load images
def preprocess_image(file_path):
    image = load_img(file_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image / 255.0  # Normalize to [0,1]
    return image

X = data['filepaths'].apply(lambda x: os.path.join('coffe_model', x)).values
y = data['class index'].values

# Load and preprocess images
X = np.array([preprocess_image(file_path) for file_path in X])
y = to_categorical(y, num_classes=4)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, validation_split=0.1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")

# Save the model as TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model to file
tflite_model_path = 'outputs/model.tflite'

# Check if the 'outputs' directory exists, and create it if it doesn't
if not os.path.exists('outputs'):
    os.makedirs('outputs')

with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
