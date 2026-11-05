import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pandas as pd
from collections import Counter
import cv2
from pathlib import Path

# Configuration
DATA_DIR = "/Users/preethi/Downloads/archive 3/Multi Cancer/Multi Cancer/Lymphoma"
CLASS_NAMES = ['lymph_cll', 'lymph_fl', 'lymph_mcl']
OUTPUT_DIR = "eda_results"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 50)
print("EXPLORATORY DATA ANALYSIS - LYMPHOMA DATASET")
print("=" * 50)

# 1. Dataset Overview
print("\n1. DATASET OVERVIEW")
print("-" * 30)

class_counts = {}
class_image_paths = {}
total_images = 0

for class_name in CLASS_NAMES:
    class_path = os.path.join(DATA_DIR, class_name)
    if os.path.exists(class_path):
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
        class_counts[class_name] = len(images)
        class_image_paths[class_name] = [os.path.join(class_path, img) for img in images]
        total_images += len(images)
        print(f"{class_name}: {len(images)} images")
    else:
        print(f"{class_name}: Directory not found!")
        class_counts[class_name] = 0
        class_image_paths[class_name] = []

print(f"\nTotal images: {total_images}")
print(f"Number of classes: {len(CLASS_NAMES)}")

# 2. Class Distribution Analysis
print("\n2. CLASS DISTRIBUTION")
print("-" * 30)

plt.figure(figsize=(10, 6))
bars = plt.bar(CLASS_NAMES, [class_counts[name] for name in CLASS_NAMES], color=['#3498db', '#e74c3c', '#2ecc71'])
plt.title('Class Distribution in Lymphoma Dataset', fontsize=14, fontweight='bold')
plt.xlabel('Lymphoma Types', fontsize=12)
plt.ylabel('Number of Images', fontsize=12)
plt.grid(axis='y', alpha=0.3)

# Add count labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'class_distribution.png'), dpi=300, bbox_inches='tight')
print("✓ Class distribution plot saved")

# Create pie chart
plt.figure(figsize=(8, 8))
colors = ['#3498db', '#e74c3c', '#2ecc71']
plt.pie([class_counts[name] for name in CLASS_NAMES], 
        labels=CLASS_NAMES, 
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        explode=(0.05, 0.05, 0.05))
plt.title('Class Distribution Percentage', fontsize=14, fontweight='bold')
plt.axis('equal')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'class_distribution_pie.png'), dpi=300, bbox_inches='tight')
print("✓ Class distribution pie chart saved")

# 3. Image Dimensions Analysis
print("\n3. IMAGE DIMENSIONS ANALYSIS")
print("-" * 30)

image_dimensions = []
aspect_ratios = []
file_sizes = []

for class_name in CLASS_NAMES:
    for img_path in class_image_paths[class_name][:100]:  # Sample 100 images per class
        try:
            img = Image.open(img_path)
            width, height = img.size
            image_dimensions.append((width, height))
            aspect_ratios.append(width / height)
            file_sizes.append(os.path.getsize(img_path) / 1024)  # KB
        except Exception as e:
            continue

if image_dimensions:
    widths = [dim[0] for dim in image_dimensions]
    heights = [dim[1] for dim in image_dimensions]
    
    print(f"Width - Mean: {np.mean(widths):.1f}, Std: {np.std(widths):.1f}, Min: {min(widths)}, Max: {max(widths)}")
    print(f"Height - Mean: {np.mean(heights):.1f}, Std: {np.std(heights):.1f}, Min: {min(heights)}, Max: {max(heights)}")
    print(f"Aspect Ratio - Mean: {np.mean(aspect_ratios):.2f}, Std: {np.std(aspect_ratios):.2f}")
    print(f"File Size - Mean: {np.mean(file_sizes):.1f} KB, Std: {np.std(file_sizes):.1f} KB")
    
    # Plot image dimensions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].scatter(widths, heights, alpha=0.5, s=50)
    axes[0].set_xlabel('Width (pixels)', fontsize=12)
    axes[0].set_ylabel('Height (pixels)', fontsize=12)
    axes[0].set_title('Image Dimensions Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(aspect_ratios, bins=30, color='purple', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Aspect Ratio (Width/Height)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Aspect Ratio Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'image_dimensions.png'), dpi=300, bbox_inches='tight')
    print("✓ Image dimensions analysis saved")

# 4. Sample Images Visualization
print("\n4. SAMPLE IMAGES VISUALIZATION")
print("-" * 30)

fig, axes = plt.subplots(3, 5, figsize=(15, 9))
fig.suptitle('Sample Images from Each Class', fontsize=16, fontweight='bold')

for class_idx, class_name in enumerate(CLASS_NAMES):
    sample_images = class_image_paths[class_name][:5]
    for img_idx, img_path in enumerate(sample_images):
        try:
            img = Image.open(img_path)
            axes[class_idx, img_idx].imshow(img)
            axes[class_idx, img_idx].axis('off')
            if img_idx == 0:
                axes[class_idx, img_idx].set_ylabel(class_name, fontsize=11, fontweight='bold')
        except Exception as e:
            axes[class_idx, img_idx].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'sample_images.png'), dpi=300, bbox_inches='tight')
print("✓ Sample images visualization saved")

# 5. Color Distribution Analysis
print("\n5. COLOR DISTRIBUTION ANALYSIS")
print("-" * 30)

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Color Channel Distributions by Class', fontsize=16, fontweight='bold')

colors = ['red', 'green', 'blue']
channel_names = ['Red', 'Green', 'Blue']

for class_idx, class_name in enumerate(CLASS_NAMES):
    # Sample images for color analysis
    sample_images = class_image_paths[class_name][:20]
    red_values = []
    green_values = []
    blue_values = []
    
    for img_path in sample_images:
        try:
            img = Image.open(img_path)
            img_array = np.array(img)
            if len(img_array.shape) == 3:  # RGB
                red_values.extend(img_array[:, :, 0].flatten())
                green_values.extend(img_array[:, :, 1].flatten())
                blue_values.extend(img_array[:, :, 2].flatten())
        except Exception as e:
            continue
    
    # Plot histograms for each channel
    for channel_idx, (values, color, name) in enumerate(zip([red_values, green_values, blue_values], colors, channel_names)):
        if values:
            axes[class_idx, channel_idx].hist(values, bins=50, color=color, alpha=0.7, density=True)
            axes[class_idx, channel_idx].set_title(f'{class_name} - {name} Channel', fontsize=10, fontweight='bold')
            axes[class_idx, channel_idx].set_xlabel('Pixel Intensity', fontsize=9)
            axes[class_idx, channel_idx].set_ylabel('Density', fontsize=9)
            axes[class_idx, channel_idx].grid(axis='y', alpha=0.3)
            
            # Add statistics
            mean_val = np.mean(values)
            std_val = np.std(values)
            axes[class_idx, channel_idx].axvline(mean_val, color='black', linestyle='--', linewidth=2)
            axes[class_idx, channel_idx].text(0.98, 0.95, f'μ={mean_val:.1f}\nσ={std_val:.1f}', 
                                           transform=axes[class_idx, channel_idx].transAxes,
                                           ha='right', va='top', fontsize=8,
                                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'color_distribution.png'), dpi=300, bbox_inches='tight')
print("✓ Color distribution analysis saved")

# 6. Statistical Summary
print("\n6. STATISTICAL SUMMARY")
print("-" * 30)

summary_data = {
    'Class': CLASS_NAMES,
    'Count': [class_counts[name] for name in CLASS_NAMES],
    'Percentage': [f"{(class_counts[name]/total_images)*100:.1f}%" for name in CLASS_NAMES]
}

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Save summary to CSV
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'dataset_summary.csv'), index=False)
print("✓ Statistical summary saved to CSV")

# 7. Image Quality Analysis
print("\n7. IMAGE QUALITY ANALYSIS")
print("-" * 30)

quality_metrics = {'blur_scores': [], 'brightness_scores': [], 'contrast_scores': []}

for class_name in CLASS_NAMES:
    class_blur = []
    class_brightness = []
    class_contrast = []
    
    for img_path in class_image_paths[class_name][:30]:  # Sample 30 images per class
        try:
            img = cv2.imread(img_path)
            if img is not None:
                # Blur score using Laplacian variance
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                class_blur.append(blur_score)
                
                # Brightness (mean intensity)
                brightness = np.mean(gray)
                class_brightness.append(brightness)
                
                # Contrast (standard deviation)
                contrast = np.std(gray)
                class_contrast.append(contrast)
        except Exception as e:
            continue
    
    quality_metrics['blur_scores'].append(class_blur)
    quality_metrics['brightness_scores'].append(class_brightness)
    quality_metrics['contrast_scores'].append(class_contrast)

# Plot quality metrics
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Blur scores
axes[0].boxplot([quality_metrics['blur_scores'][i] for i in range(len(CLASS_NAMES))], 
                labels=CLASS_NAMES)
axes[0].set_title('Image Sharpness (Blur Score)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Laplacian Variance', fontsize=11)
axes[0].grid(axis='y', alpha=0.3)

# Brightness
axes[1].boxplot([quality_metrics['brightness_scores'][i] for i in range(len(CLASS_NAMES))], 
                labels=CLASS_NAMES)
axes[1].set_title('Image Brightness', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Mean Pixel Intensity', fontsize=11)
axes[1].grid(axis='y', alpha=0.3)

# Contrast
axes[2].boxplot([quality_metrics['contrast_scores'][i] for i in range(len(CLASS_NAMES))], 
                labels=CLASS_NAMES)
axes[2].set_title('Image Contrast', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Standard Deviation', fontsize=11)
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'image_quality.png'), dpi=300, bbox_inches='tight')
print("✓ Image quality analysis saved")

print("\n" + "=" * 50)
print("EDA ANALYSIS COMPLETED!")
print("=" * 50)
print(f"All results saved to: {OUTPUT_DIR}/")
print("Generated files:")
print("  - class_distribution.png")
print("  - class_distribution_pie.png")
print("  - image_dimensions.png")
print("  - sample_images.png")
print("  - color_distribution.png")
print("  - dataset_summary.csv")
print("  - image_quality.png")
