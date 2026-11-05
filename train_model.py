import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import json

# Configuration
DATA_DIR = "/Users/preethi/Downloads/archive 3/Multi Cancer/Multi Cancer/Lymphoma"
IMG_SIZE = (128, 128)  # Smaller for faster training
BATCH_SIZE = 32  # Reduced batch size for hybrid model (more memory intensive)
EPOCHS = 5  # Increased epochs for hybrid model convergence
LEARNING_RATE = 0.001
NUM_CLASSES = 3
CLASS_NAMES = ['lymph_cll', 'lymph_fl', 'lymph_mcl']
MODEL_TYPE = "Hybrid Deep Learning with Multi-scale Feature Fusion"

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

print("Setting up data generators...")

# Data augmentation for training (increased to reduce overfitting)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    zoom_range=0.3,
    shear_range=0.2,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)

# Only rescaling for validation
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Load training data
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    classes=CLASS_NAMES
)

# Load validation data
validation_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    classes=CLASS_NAMES
)

print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")
print(f"Class indices: {train_generator.class_indices}")
print(f"Model Type: {MODEL_TYPE}")

# Build Hybrid Deep Learning Model with Multi-scale Feature Fusion
print("Building hybrid model with MobileNetV2 and DenseNet121...")

# Input layer
inputs = keras.Input(shape=IMG_SIZE + (3,))

# MobileNetV2 backbone
mobilenet_base = keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=IMG_SIZE + (3,)
)
mobilenet_base.trainable = False

# DenseNet121 backbone
densenet_base = keras.applications.DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=IMG_SIZE + (3,)
)
densenet_base.trainable = False

# Multi-scale feature extraction from MobileNetV2
mobilenet_features = mobilenet_base(inputs, training=False)
# Extract features from different scales using different pooling strategies
mobilenet_low = layers.GlobalAveragePooling2D()(mobilenet_features)  # Global features
mobilenet_mid = layers.MaxPooling2D(pool_size=(2, 2))(mobilenet_features)  # Mid-level features
mobilenet_mid = layers.GlobalAveragePooling2D()(mobilenet_mid)
mobilenet_high = layers.AveragePooling2D(pool_size=(4, 4))(mobilenet_features)  # High-level features
mobilenet_high = layers.GlobalAveragePooling2D()(mobilenet_high)

# Multi-scale feature extraction from DenseNet121
densenet_features = densenet_base(inputs, training=False)
# Extract features from different scales using different pooling strategies
densenet_low = layers.GlobalAveragePooling2D()(densenet_features)   # Global features
densenet_mid = layers.MaxPooling2D(pool_size=(2, 2))(densenet_features)    # Mid-level features
densenet_mid = layers.GlobalAveragePooling2D()(densenet_mid)
densenet_high = layers.AveragePooling2D(pool_size=(4, 4))(densenet_features)   # High-level features
densenet_high = layers.GlobalAveragePooling2D()(densenet_high)

# Add attention mechanism for feature fusion
class AttentionFusion(layers.Layer):
    def __init__(self, units=256):
        super(AttentionFusion, self).__init__()
        self.units = units
        
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.units),
                                initializer='glorot_uniform',
                                trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                initializer='zeros',
                                trainable=True)
        self.u = self.add_weight(shape=(self.units, 1),
                                initializer='glorot_uniform',
                                trainable=True)
        
    def call(self, inputs):
        # inputs shape: (batch, features)
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
        weighted_inputs = inputs * attention_weights
        return tf.reduce_sum(weighted_inputs, axis=1)

# Concatenate multi-scale features from both backbones
combined_features = layers.Concatenate()([
    mobilenet_low, mobilenet_mid, mobilenet_high,
    densenet_low, densenet_mid, densenet_high
])

# Apply attention fusion
attention_fusion = AttentionFusion(units=256)
fused_features = attention_fusion(combined_features)

# Add batch normalization and dimensionality reduction
x = layers.BatchNormalization()(fused_features)
x = layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = keras.Model(inputs, outputs)

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

print("Model summary:")
model.summary()

# Callbacks with more aggressive early stopping
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        min_delta=0.001
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    ),
    keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True
    )
]

# Train model
print("\nStarting training...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

# Fine-tune both base models (conservative approach to prevent overfitting)
print("\nFine-tuning hybrid model...")
# Only unfreeze the last few layers of both backbones
mobilenet_base.trainable = True
densenet_base.trainable = True

# Freeze all layers except the last 20 of MobileNetV2
for layer in mobilenet_base.layers[:-20]:
    layer.trainable = False

# Freeze all layers except the last 20 of DenseNet121
for layer in densenet_base.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE/100),  # Much lower learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

history_fine = model.fit(
    train_generator,
    epochs=5,  # Increased epochs for hybrid model fine-tuning
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

# Save the final model
print("\nSaving model...")
model.save('lymphoma_classifier.keras')
print("Model saved as 'lymphoma_classifier.keras'")

# Save class indices
class_indices = train_generator.class_indices
with open('class_indices.json', 'w') as f:
    json.dump(class_indices, f)
print("Class indices saved as 'class_indices.json'")

# Evaluate model
print("\nEvaluating model...")
test_loss, test_acc, test_auc = model.evaluate(validation_generator)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test AUC: {test_auc:.4f}")

# Convert to TensorFlow.js format for web deployment
print("\nConverting to TensorFlow.js format...")
try:
    import tensorflowjs as tfjs
    tfjs.converters.save_keras_model(
        model,
        'tfjs_model',
        weight_format_bin=True
    )
    print("Model converted and saved in 'tfjs_model' directory")
except ImportError:
    print("TensorFlow.js not installed. Install with: pip install tensorflowjs")
    print("Model saved as Keras format only")

print("\nTraining completed!")
