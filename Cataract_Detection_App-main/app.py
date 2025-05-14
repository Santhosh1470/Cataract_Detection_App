import os
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Set the upload folder
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load the trained cataract detection model
MODEL_PATH = "model/vgg19_model_final.h5"  # Update this if your model is in a different location
model = load_model(MODEL_PATH)

# Function to process image and make prediction
def predict_cataract(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize image to match model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize pixel values

    prediction = model.predict(img_array)
    return "Cataract Detected" if prediction[0][0] > 0.5 else "No Cataract"

# Route for home page
@app.route("/")
def home():
    return render_template("index.html")  # Home page with "Get Started" button

# Route for detection page
@app.route("/detect")
def detect_page():
    return render_template("detect.html")  # Upload page

# Route to handle image upload and prediction
@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"})

    # Save uploaded image
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Make prediction
    result = predict_cataract(file_path)

    return jsonify({"prediction": result, "image_url": f"/uploads/{file.filename}"})

# Route to serve uploaded images
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
