from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("TuberModel.keras")

# Ensure the 'static/uploads' directory exists
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create folder if it doesn’t exist

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Function to preprocess image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to model input size
    img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)  # Save uploaded file
            
            # Preprocess and predict
            img_array = preprocess_image(file_path)
            prediction = model.predict(img_array)[0][0]
            
            # Convert prediction to readable format
            result = "Tuberculosis Detected" if prediction > 0.5 else "Normal"
            confidence = f"{prediction * 100:.2f}%"
            
            return render_template("index.html", result=result, confidence=confidence, image_path=file_path)

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
