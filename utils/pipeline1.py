# Import necessary libraries
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

    def create_pipeline(self, image_directory):
        """Create a batched and prefetched dataset from image directory"""
        dataset = tf.data.Dataset.list_files(
            f"{image_directory}/*.jpg",
            shuffle=True,
            seed=self.seed
        )
        dataset = dataset.map(
            self.load_and_preprocess_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

# Example usage:
# pipeline = ImagePipeline()
# image_pipeline = pipeline.create_pipeline("path_to_your_image_directory")
# for batch in image_pipeline.take(1):
#     print(batch.shape)  # Should print (batch_size, 256, 256, 3)