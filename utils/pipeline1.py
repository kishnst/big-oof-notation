# Import necessary libraries
import tensorflow as tf

# Define constants
seed = 123
image_size = (256, 256)
batch_size = 32

# Function to preprocess images
def preprocess_image(image):
    # Decode the image
    image = tf.image.decode_jpeg(image, channels=3)
    # Resize the image to the target size
    image = tf.image.resize(image, image_size)
    # Normalize the image to [0, 1]
    image = image / 255.0
    return image

# Function to load and preprocess images from file paths
def load_and_preprocess_image(file_path):
    # Read the image file
    image = tf.io.read_file(file_path)
    # Preprocess the image
    return preprocess_image(image)

# Create a dataset from a directory of images
def create_image_pipeline(image_directory):
    # Load file paths from the directory
    dataset = tf.data.Dataset.list_files(f"{image_directory}/*.jpg", shuffle=True, seed=seed)
    # Load and preprocess images
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    # Batch the dataset
    dataset = dataset.batch(batch_size)
    # Prefetch to improve performance
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# Example usage
# image_directory = "path_to_your_image_directory"  # Replace with your image directory path
# image_pipeline = create_image_pipeline(image_directory)

# Iterate through the dataset (for demonstration purposes)
# for batch in image_pipeline.take(1):
    # print(batch.shape)  # Should print (batch_size, 256, 256, 3)