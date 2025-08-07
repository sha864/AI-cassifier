# ✅ 1. Import Libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# ✅ 2. Download and Prepare Dataset (Cats vs Dogs)
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs_filtered.zip', origin=_URL, extract=True)
base_dir = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# ✅ 3. Image Preprocessing
IMG_SIZE = 160  # MobileNetV2 input size
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# ✅ 4. Load Pretrained Model (MobileNetV2)
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                         include_top=False,
                         weights='imagenet')

base_model.trainable = False  # Freeze base model

# ✅ 5. Add Custom Layers on Top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)  # Binary classification

model = Model(inputs=base_model.input, outputs=predictions)

# ✅ 6. Compile the Model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ✅ 7. Train the Model
EPOCHS = 5
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS
)

# ✅ 8. Evaluate the Model
acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
print(f"\n✅ Final Training Accuracy: {acc:.2f}")
print(f"✅ Final Validation Accuracy: {val_acc:.2f}")

# ✅ 9. Plot Accuracy & Loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# ✅ 10. Display Sample Predictions
sample_batch, labels = next(validation_generator)
sample_images = sample_batch[:5]
sample_labels = labels[:5]

preds = model.predict(sample_images)

plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(sample_images[i])
    label = "Dog" if preds[i] > 0.5 else "Cat"
    plt.title(f"Pred: {label}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# ✅ 11. Save the Trained Model (optional)
model.save("cat_dog_classifier_mobilenetv2.h5")
