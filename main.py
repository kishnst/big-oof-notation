from flask import Flask, request, jsonify


import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
from flask_cors import CORS
CORS(app)

# Load your model
model = tf.keras.models.load_model("model/densenet_try_three_keras_with_co.keras")

class ImagePipeline:
    def __init__(self, seed=123, image_size=(256, 256), batch_size=32):
        self.seed = seed
        self.image_size = image_size
        self.batch_size = batch_size
    
    def load_and_preprocess_image(self, file_path):
        """Read and preprocess an image from file path"""
        image = tf.io.read_file(file_path)
        return self.preprocess_image(image)

# Image preprocessing
def preprocess_image(image_bytes):
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, (256, 256))
    image = image / 255.0
    return tf.expand_dims(image, axis=0)

def preprocess_image(image_bytes):
         image = tf.io.decode_jpeg(tf.convert_to_tensor(image_bytes), channels=3)
         image = tf.image.resize(image, (256, 256))
         image = image / 255.0
         return tf.expand_dims(image, axis=0)

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file found"}), 400

    try:
        image_bytes = request.files['image'].read()
        processed_image = preprocess_image(image_bytes)
        predictions = model.predict(processed_image)
        predicted_class = int(tf.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)