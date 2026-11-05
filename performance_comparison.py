import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import time
from datetime import datetime
import json

# Configuration
DATA_DIR = "/Users/preethi/Downloads/archive 3/Multi Cancer/Multi Cancer/Lymphoma"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 5  # Reduced for comparison purposes
LEARNING_RATE = 0.001
NUM_CLASSES = 3
CLASS_NAMES = ['lymph_cll', 'lymph_fl', 'lymph_mcl']

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

print("=" * 60)
print("PERFORMANCE COMPARISON - LYMPHOMA CLASSIFICATION MODELS")
print("=" * 60)

# Data preparation
print("\nPreparing data...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    classes=CLASS_NAMES
)

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

# Model architectures to compare
def build_mobilenet_model():
    """MobileNetV2 only model"""
    base_model = keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=IMG_SIZE + (3,)
    )
    base_model.trainable = False
    
    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model, "MobileNetV2"

def build_densenet_model():
    """DenseNet121 only model"""
    base_model = keras.applications.DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=IMG_SIZE + (3,)
    )
    base_model.trainable = False
    
    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model, "DenseNet121"

def build_hybrid_model():
    """Hybrid MobileNetV2 + DenseNet121 with multi-scale features"""
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
    
    # Input layer
    inputs = keras.Input(shape=IMG_SIZE + (3,))
    
    # Multi-scale feature extraction from MobileNetV2
    mobilenet_features = mobilenet_base(inputs, training=False)
    mobilenet_low = layers.GlobalAveragePooling2D()(mobilenet_features)
    mobilenet_mid = layers.MaxPooling2D(pool_size=(2, 2))(mobilenet_features)
    mobilenet_mid = layers.GlobalAveragePooling2D()(mobilenet_mid)
    mobilenet_high = layers.AveragePooling2D(pool_size=(4, 4))(mobilenet_features)
    mobilenet_high = layers.GlobalAveragePooling2D()(mobilenet_high)
    
    # Multi-scale feature extraction from DenseNet121
    densenet_features = densenet_base(inputs, training=False)
    densenet_low = layers.GlobalAveragePooling2D()(densenet_features)
    densenet_mid = layers.MaxPooling2D(pool_size=(2, 2))(densenet_features)
    densenet_mid = layers.GlobalAveragePooling2D()(densenet_mid)
    densenet_high = layers.AveragePooling2D(pool_size=(4, 4))(densenet_features)
    densenet_high = layers.GlobalAveragePooling2D()(densenet_high)
    
    # Concatenate multi-scale features
    combined_features = layers.Concatenate()([
        mobilenet_low, mobilenet_mid, mobilenet_high,
        densenet_low, densenet_mid, densenet_high
    ])
    
    # Classification head
    x = layers.BatchNormalization()(combined_features)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model, "Hybrid MobileNetV2 + DenseNet121"

# Models to compare
models_to_test = [
    build_mobilenet_model,
    build_densenet_model,
    build_hybrid_model
]

# Store results
results = []

for model_builder in models_to_test:
    print(f"\n{'='*60}")
    model, model_name = model_builder()
    print(f"Training: {model_name}")
    print(f"{'='*60}")
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc'), keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')]
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            min_delta=0.001
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
    ]
    
    # Train model
    start_time = time.time()
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    training_time = time.time() - start_time
    
    # Evaluate model
    print(f"\nEvaluating {model_name}...")
    eval_start = time.time()
    test_loss, test_acc, test_auc, test_precision, test_recall = model.evaluate(validation_generator)
    eval_time = time.time() - eval_start
    
    # Calculate F1 score
    f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-7)
    
    # Get model parameters count
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    
    # Store results
    result = {
        'Model': model_name,
        'Accuracy': f"{test_acc:.4f}",
        'AUC': f"{test_auc:.4f}",
        'Precision': f"{test_precision:.4f}",
        'Recall': f"{test_recall:.4f}",
        'F1-Score': f"{f1_score:.4f}",
        'Training Time (s)': f"{training_time:.2f}",
        'Inference Time (s)': f"{eval_time:.2f}",
        'Total Parameters': f"{total_params:,}",
        'Trainable Parameters': f"{trainable_params:,}"
    }
    results.append(result)
    
    print(f"\n{model_name} Results:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  AUC: {test_auc:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  F1-Score: {f1_score:.4f}")
    print(f"  Training Time: {training_time:.2f}s")
    print(f"  Total Parameters: {total_params:,}")

# Create comparison table
print(f"\n{'='*60}")
print("PERFORMANCE COMPARISON TABLE")
print(f"{'='*60}")

df = pd.DataFrame(results)

# Display table
print("\n" + df.to_string(index=False))

# Save to CSV
df.to_csv('performance_comparison.csv', index=False)
print("\n✓ Comparison table saved to 'performance_comparison.csv'")

# Create formatted table for research paper
print(f"\n{'='*60}")
print("FORMATTED TABLE FOR RESEARCH PAPER")
print(f"{'='*60}")

# Create LaTeX-style table
latex_table = r"""
\begin{table}[h]
\centering
\caption{Performance Comparison of Different Model Architectures for Lymphoma Classification}
\label{tab:performance_comparison}
\begin{tabular}{|l|c|c|c|c|c|c|c|}
\hline
\textbf{Model} & \textbf{Accuracy} & \textbf{AUC} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} & \textbf{Training Time (s)} & \textbf{Parameters} \\
\hline
"""

for result in results:
    latex_table += f"{result['Model']} & {result['Accuracy']} & {result['AUC']} & {result['Precision']} & {result['Recall']} & {result['F1-Score']} & {result['Training Time (s)']} & {result['Total Parameters']} \\\\\n"

latex_table += r"""\hline
\end{tabular}
\end{table}
"""

print(latex_table)

# Save LaTeX table
with open('performance_comparison_latex.txt', 'w') as f:
    f.write(latex_table)
print("✓ LaTeX table saved to 'performance_comparison_latex.txt'")

# Create markdown table
markdown_table = """
## Performance Comparison Table

| Model | Accuracy | AUC | Precision | Recall | F1-Score | Training Time (s) | Parameters |
|-------|----------|-----|-----------|--------|----------|-------------------|------------|
"""

for result in results:
    markdown_table += f"| {result['Model']} | {result['Accuracy']} | {result['AUC']} | {result['Precision']} | {result['Recall']} | {result['F1-Score']} | {result['Training Time (s)']} | {result['Total Parameters']} |\n"

# Save markdown table
with open('performance_comparison_markdown.md', 'w') as f:
    f.write(markdown_table)
print("✓ Markdown table saved to 'performance_comparison_markdown.md'")

print(f"\n{'='*60}")
print("COMPARISON COMPLETE!")
print(f"{'='*60}")
print("Generated files:")
print("  - performance_comparison.csv")
print("  - performance_comparison_latex.txt")
print("  - performance_comparison_markdown.md")
