from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import json

app = Flask(__name__)

# Load your trained model
MODEL_PATH = "plant_disease_prediction_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Load class indices
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Convert keys to int so we can use them directly
classes = {int(k): v for k, v in class_indices.items()}

# ✅ Preprocessing function
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)              # Load
    img = img.resize(target_size)             # Resize to model input size
    img_array = np.array(img)                 # Convert to numpy array
    img_array = np.expand_dims(img_array, 0)  # Add batch dimension
    img_array = img_array.astype('float32') / 255.0  # Normalize [0,1]
    return img_array


@app.route('/predict', methods=['GET'])
def home():
    return render_template('index.html')



@app.route("/dashboard")
def dashboard():
    return render_template("Dashboard.html")

@app.route("/croprecommendation")
def croprecommendation():
    return render_template("Croprecommendation.html")

@app.route('/' , methods=['GET'])
def homepage():
    return render_template("Homepage.html")

@app.route("/loginpage")
def loginpage():
    return render_template("Loginpage.html")

@app.route("/prices")
def prices():
    return render_template("Prices.html")

@app.route("/profile")
def profile():
    return render_template("Profile.html")

@app.route("/signup")
def signup():
    return render_template("Signup.html")

@app.route("/weather")
def weather():
    return render_template("Weather.html")



@app.route('/', methods=['POST'])
def predict():
    # Save uploaded file
    imagefile = request.files['imagefile']
    image_path = os.path.join("images", imagefile.filename)
    imagefile.save(image_path)

    # Preprocess and predict
    img_array = load_and_preprocess_image(image_path, target_size=(224, 224))
    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction, axis=1)[0]
    class_name = classes[predicted_class]
    confidence = round(100 * np.max(prediction), 2)

    result = f"{class_name} ({confidence}%)"

    return render_template('index.html', prediction=result)


if __name__ == '__main__':
    app.run(port=3000, debug=True)
