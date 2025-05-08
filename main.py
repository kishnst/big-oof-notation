import tensorflow as tf

class ImagePipeline:
    def __init__(self, seed=123, image_size=(256, 256), batch_size=32):
        self.seed = seed
        self.image_size = image_size
        self.batch_size = batch_size

    def preprocess_image(self, image):
        """Decode, resize and normalize an image"""
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.image_size)
        image = image / 255.0
        return image
    
    def load_and_preprocess_image(self, file_path):
        """Read and preprocess an image from file path"""
        image = tf.io.read_file(file_path)
        return self.preprocess_image(image)


pipeline = ImagePipeline()
input_image = "/Users/kishanthayyil/Downloads/pot.jpg"  # Replace with actual image path
preprocessed_image = pipeline.load_and_preprocess_image(input_image)

model = tf.keras.models.load_model("/Users/kishanthayyil/Python files/big-oof-notation/model/densenet_try_three_keras_with_co.keras")
# Add batch dimension to match model input shape
input_batch = tf.expand_dims(preprocessed_image, axis=0)

# Get model predictions
predictions = model.predict(input_batch)

# Get the predicted class index
predicted_class = tf.argmax(predictions[0]).numpy()

# Print the prediction results
print("Model predictions:", predictions)
print("Predicted class index:", predicted_class)



