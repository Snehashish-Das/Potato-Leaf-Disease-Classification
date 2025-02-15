from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['PROCESSED_FOLDER'] = 'static/processed/'  # Folder for segmented images
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load the models
model_full = load_model('potatoCNN_kerasmodel.keras')  # Model A (Full Image)
model_parts = load_model('final_potato_leaf_model.h5')  # Model B (Segmented Parts)

# Class labels
class_names = ['Early blight', 'Late blight', 'Healthy']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def segment_image(img_path):
    """Divides the input image into 5 regions: top-left, top-right, bottom-left, bottom-right, and center."""
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    cx, cy = w // 2, h // 2  # Center coordinates

    segmented_paths = []

    # Define the 5 regions
    regions = {
        "top_left": img[0:cy, 0:cx],
        "top_right": img[0:cy, cx:w],
        "bottom_left": img[cy:h, 0:cx],
        "bottom_right": img[cy:h, cx:w],
        "center": img[cy//2:cy + cy//2, cx//2:cx + cx//2]  # Small center region
    }

    # Save each segmented part
    for name, part in regions.items():
        part_filename = f"{name}.jpg"
        part_path = os.path.join(app.config['PROCESSED_FOLDER'], part_filename)
        cv2.imwrite(part_path, part)
        segmented_paths.append(part_path)

    return segmented_paths

def predict_image(img_path, model):
    """Predicts the class of an image using the specified model."""
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0).astype("float32")

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    
    return predicted_class, confidence

def predict_parts(segmented_paths):
    """Predicts the class of each segmented part using Model B."""
    part_results = []
    for part_path in segmented_paths:
        pred_class, conf = predict_image(part_path, model_parts)
        part_results.append((part_path, pred_class, conf))
    return part_results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Full image prediction (Model A)
        predicted_class, confidence = predict_image(filepath, model_full)

        # 5-segment processing (Model B)
        segmented_paths = segment_image(filepath)
        part_predictions = predict_parts(segmented_paths)

        return render_template(
            'index.html', 
            filename=filename, 
            predicted_class=predicted_class, 
            confidence=confidence,
            segmented_paths=segmented_paths,
            part_predictions=part_predictions
        )
    
    return redirect(request.url)

if __name__ == '__main__':
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)  # Ensure processed folder exists
    app.run(debug=True)
