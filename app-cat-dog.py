import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder="templates")

# Define the path to the model file
model_file = 'catdogclassifier.h5'

# Check if the model file exists
if os.path.exists(model_file):
    # Load your pre-trained model
    model = load_model(model_file)
else:
    print(f"Error: Model file '{model_file}' not found. Please check the file path.")
    model = None  # Set model to None if the file is not found

# Function to preprocess and predict the image
def predict_image(image, threshold_low=0.3, threshold_high=0.8):
    if model is None:
        return {"error": "Model not loaded"}
    
    # Resize image to model input size
    image = cv2.resize(image, (256, 256))
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Get the prediction from the model
    prediction = model.predict(image)
    pred_value = prediction[0][0]  # Assuming a binary classification with a single output node

    # Classify the result based on thresholds
    if pred_value > threshold_high:
        return {"class": "Dog", "confidence": float(pred_value)}
    elif pred_value < threshold_low:
        return {"class": "Cat", "confidence": float(1 - pred_value)}
    else:
        return {"class": "Neither", "confidence": float(pred_value)}

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    image_np = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    
    if image is None:
        return jsonify({"error": "Invalid image file"}), 400
    
    result = predict_image(image)
    
    return render_template("result.html", result=result)

if __name__ == '__main__':
    app.run(debug=True, port=5600)
