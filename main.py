# from flask import Flask, request, jsonify
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import io

# app = Flask(__name__)

# class ImagePipeline:
#     def __init__(self, seed=123, image_size=(256, 256), batch_size=32):
#         self.seed = seed
#         self.image_size = image_size
#         self.batch_size = batch_size

#     def preprocess_image(self, image):
#         """Decode, resize and normalize an image"""
#         image = tf.image.decode_jpeg(image, channels=3)
#         image = tf.image.resize(image, self.image_size)
#         image = image / 255.0
#         return image
    
#     def load_and_preprocess_image(self, file_path):
#         """Read and preprocess an image from file path"""
#         image = tf.io.read_file(file_path)
#         return self.preprocess_image(image)


# # Load model once
# model = tf.keras.models.load_model("C:/Users/vishw/Desktop/hackthon/densenet_try_three_keras_with_co.keras")
# pipeline = ImagePipeline()

# def preprocess_uploaded_image(image_bytes):
#     image = tf.image.decode_jpeg(image_bytes, channels=3)
#     image = tf.image.resize(image, (256, 256))
#     image = image / 255.0
#     image = tf.expand_dims(image, axis=0)
#     return image

# @app.route("/predict", methods=["POST"])
# def predict():
#     if 'image' not in request.files:
#         return jsonify({"error": "No image uploaded"}), 400

#     file = request.files['image']
#     image_bytes = file.read()

#     try:
#         processed = preprocess_uploaded_image(image_bytes)
#         predictions = model.predict(processed)
#         predicted_class = int(tf.argmax(predictions[0]).numpy())

#         return jsonify({
#             "predicted_class": predicted_class,
#             "confidence": float(np.max(predictions[0]))
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)


from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load your model
model = tf.keras.models.load_model("C:/Users/vishw/Desktop/hackthon/densenet_try_three_keras_with_co.keras")

# Image preprocessing
def preprocess_image(image_bytes):
    image = tf.image.decode_jpeg(image_bytes, channels=3)
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
    app.run(host="0.0.0.0", port=5000)
