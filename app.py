from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load your trained model
model_path = './model/best_model.hdf5'  # Adjust path as needed
model = load_model(model_path)

def prepare_image(image, target_size):
    """Prepare the image for prediction."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.asarray(image)
    image = (image.astype(np.float32) / 255.0)  # Normalize the image
    return np.expand_dims(image, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    
    file = request.files['image']
    filename = secure_filename(file.filename)  # You can save the file to the server if needed
    image = Image.open(file.stream)  # Open the image directly from the file stream

    # Adjust target_size as per your model's requirement
    processed_image = prepare_image(image, target_size=(96, 96))  

    # Predict
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Adjust according to how your model's output is structured

    # Map the numerical prediction to a human-readable category
    categories = ['pip', 'handel_door', 'power_socket']  # Example categories, replace with your actual categories
    category = categories[predicted_class]

    return jsonify({'category': category})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
