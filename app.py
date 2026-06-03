import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import json
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Disease classes
DISEASE_CLASSES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Corn___Cercospora_leaf_spot',
    'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight',
    'Corn___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___healthy'
]

DISEASE_INFO = {
    'Apple___Apple_scab': {
        'name': 'Apple Scab',
        'severity': 'Moderate',
        'description': 'Fungal disease causing dark, scabby lesions on leaves and fruit.',
        'treatment': 'Apply fungicide sprays early in the season. Remove fallen leaves to reduce spore sources.',
        'color': '#e74c3c'
    },
    'Apple___Black_rot': {
        'name': 'Black Rot',
        'severity': 'High',
        'description': 'Fungal infection causing circular brown lesions with purple borders.',
        'treatment': 'Prune infected branches. Apply copper-based fungicides. Improve air circulation.',
        'color': '#c0392b'
    },
    'Apple___Cedar_apple_rust': {
        'name': 'Cedar Apple Rust',
        'severity': 'Moderate',
        'description': 'Fungal disease creating bright orange spots on leaves.',
        'treatment': 'Remove nearby cedar trees if possible. Apply preventive fungicide in spring.',
        'color': '#e67e22'
    },
    'Apple___healthy': {
        'name': 'Healthy Apple',
        'severity': 'None',
        'description': 'The plant appears to be in excellent health with no visible disease symptoms.',
        'treatment': 'Continue regular care and monitoring to maintain plant health.',
        'color': '#27ae60'
    },
    'Corn___Cercospora_leaf_spot': {
        'name': 'Cercospora Leaf Spot',
        'severity': 'Moderate',
        'description': 'Fungal disease creating rectangular gray lesions on corn leaves.',
        'treatment': 'Plant resistant hybrids. Apply fungicide at early signs. Rotate crops annually.',
        'color': '#e74c3c'
    },
    'Corn___Common_rust': {
        'name': 'Common Rust',
        'severity': 'Moderate',
        'description': 'Fungal infection producing reddish-brown pustules on leaf surfaces.',
        'treatment': 'Use resistant varieties. Apply fungicide if infection is severe before tasseling.',
        'color': '#d35400'
    },
    'Corn___Northern_Leaf_Blight': {
        'name': 'Northern Leaf Blight',
        'severity': 'High',
        'description': 'Fungal disease causing large, cigar-shaped tan lesions on leaves.',
        'treatment': 'Plant resistant hybrids. Apply fungicide at early leaf stages.',
        'color': '#c0392b'
    },
    'Corn___healthy': {
        'name': 'Healthy Corn',
        'severity': 'None',
        'description': 'The corn plant appears healthy with no signs of disease.',
        'treatment': 'Continue regular watering, fertilization, and pest monitoring.',
        'color': '#27ae60'
    },
    'Tomato___Bacterial_spot': {
        'name': 'Bacterial Spot',
        'severity': 'High',
        'description': 'Bacterial infection causing small, water-soaked spots on leaves and fruit.',
        'treatment': 'Apply copper bactericide. Avoid overhead irrigation. Remove infected plant parts.',
        'color': '#e74c3c'
    },
    'Tomato___healthy': {
        'name': 'Healthy Tomato',
        'severity': 'None',
        'description': 'The tomato plant is in excellent condition with vibrant, disease-free leaves.',
        'treatment': 'Maintain consistent watering and nutrition. Monitor regularly for pests.',
        'color': '#27ae60'
    }
}

# Mock model for demo (replace with actual trained model)
class MockModel:
    def predict(self, img_array):
        # Simulate prediction - in production, use actual model
        np.random.seed(int(np.sum(img_array)) % 100)
        probs = np.random.dirichlet(np.ones(10) * 0.5)
        return np.array([probs])

model = MockModel()

# Uncomment to load actual model:
# model = load_model('model/plant_disease_model.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_disease(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx]) * 100
    predicted_class = DISEASE_CLASSES[predicted_class_idx]
    
    # Top 3 predictions
    top3_idx = np.argsort(predictions[0])[-3:][::-1]
    top3 = [
        {
            'disease': DISEASE_CLASSES[i],
            'name': DISEASE_INFO[DISEASE_CLASSES[i]]['name'],
            'confidence': round(float(predictions[0][i]) * 100, 2)
        }
        for i in top3_idx
    ]
    
    return predicted_class, confidence, top3

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        predicted_class, confidence, top3 = predict_disease(filepath)
        disease_info = DISEASE_INFO[predicted_class]
        
        return jsonify({
            'success': True,
            'image_url': f'/static/uploads/{filename}',
            'predicted_class': predicted_class,
            'disease_name': disease_info['name'],
            'confidence': round(confidence, 2),
            'severity': disease_info['severity'],
            'description': disease_info['description'],
            'treatment': disease_info['treatment'],
            'color': disease_info['color'],
            'top3': top3
        })
    
    return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG.'}), 400

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=5000)
