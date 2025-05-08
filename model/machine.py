from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, optimizers
import tensorflow as tf

import os
import shutil
from pathlib import Path

DATASET_PATH = "CoLeaf DATASET"


# Load and preprocess the dataset
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    label_mode="int",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(256, 256),
    batch_size=32
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    label_mode="int",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(256, 256),
    batch_size=32
)

class_names = train_dataset.class_names
print("Class names:", class_names)

# Data augmentation and normalization
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2)
])

normalization_layer = tf.keras.layers.Rescaling(1./255)

train_dataset = train_dataset.map(
    lambda x, y: (normalization_layer(data_augmentation(x, training=True)), y)
)

validation_dataset = validation_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# Use DenseNet121 instead of ResNet50
base_model = DenseNet121(input_shape=(256, 256, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer=optimizers.Adam(learning_rate=0.0005),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
model.fit(train_dataset, validation_data=validation_dataset, epochs=100, callbacks=[early_stopping])

# Save the model
model.save("densenet_try_three_keras_with_co.keras")



# # Count the number of images in the dataset
# # image_count = 0
# # for root, dirs, files in os.walk(DATASET_PATH):
# #     image_count = 0
# #     for file in files:
# #         if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
# #             image_count += 1
# #     print(dirs, image_count)
    
# # # print(f"Total number of images: {image_count}")

