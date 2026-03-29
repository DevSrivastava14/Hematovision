from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load model
model = load_model("Blood Cell.h5")

# Class labels (VERY IMPORTANT - match folder names)
classes = ['eosinophil', 'lymphocyte', 'monocyte', 'neutrophil']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']

    if file.filename == '':
        return "No file selected"

    # Save uploaded image
    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]

    return render_template('result.html', prediction=predicted_class, img_path=filepath)

if __name__ == "__main__":
    app.run(debug=True)