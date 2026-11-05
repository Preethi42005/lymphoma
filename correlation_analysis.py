import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import cv2

# Configuration
DATA_DIR = "/Users/preethi/Downloads/archive 3/Multi Cancer/Multi Cancer/Lymphoma"
CLASS_NAMES = ['lymph_cll', 'lymph_fl', 'lymph_mcl']
OUTPUT_DIR = "eda_results"

print("Generating correlation heatmap and classification visualization...")

# Extract features from images
def extract_features(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert to different color spaces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # Color statistics
        features = {
            # RGB statistics
            'red_mean': np.mean(img[:, :, 2]),
            'red_std': np.std(img[:, :, 2]),
            'green_mean': np.mean(img[:, :, 1]),
            'green_std': np.std(img[:, :, 1]),
            'blue_mean': np.mean(img[:, :, 0]),
            'blue_std': np.std(img[:, :, 0]),
            
            # Grayscale statistics
            'gray_mean': np.mean(gray),
            'gray_std': np.std(gray),
            
            # HSV statistics
            'hue_mean': np.mean(hsv[:, :, 0]),
            'saturation_mean': np.mean(hsv[:, :, 1]),
            'value_mean': np.mean(hsv[:, :, 2]),
            
            # LAB statistics
            'L_mean': np.mean(lab[:, :, 0]),
            'A_mean': np.mean(lab[:, :, 1]),
            'B_mean': np.mean(lab[:, :, 2]),
            
            # Texture features (using gradients)
            'gradient_mean': np.mean(np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0))),
            'gradient_std': np.std(np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0))),
            
            # Edge density
            'edge_density': np.sum(cv2.Canny(gray, 100, 200)) / (gray.shape[0] * gray.shape[1]),
            
            # Sharpness (Laplacian variance)
            'sharpness': cv2.Laplacian(gray, cv2.CV_64F).var(),
            
            # Brightness and contrast
            'brightness': np.mean(gray),
            'contrast': np.std(gray),
            
            # Intensity statistics
            'intensity_mean': np.mean(img),
            'intensity_std': np.std(img),
        }
        
        return features
    except Exception as e:
        return None

# Sample images from each class
all_features = []
class_labels = []

for class_name in CLASS_NAMES:
    class_path = os.path.join(DATA_DIR, class_name)
    if os.path.exists(class_path):
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
        sample_images = images[:50]  # Sample 50 images per class
        
        for img_name in sample_images:
            img_path = os.path.join(class_path, img_name)
            features = extract_features(img_path)
            if features:
                all_features.append(features)
                class_labels.append(class_name)

# Create DataFrame
df = pd.DataFrame(all_features)
df['class'] = class_labels

print(f"Extracted features from {len(df)} images")

# 1. Correlation Heatmap
print("Creating correlation heatmap...")
plt.figure(figsize=(14, 12))

# Select only numeric columns for correlation
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()

# Create heatmap
sns.heatmap(correlation_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            fmt='.2f',
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            annot_kws={'size': 8})

plt.title('Feature Correlation Heatmap - Lymphoma Image Dataset', 
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
print("✓ Correlation heatmap saved")

# 2. Classification Visualization (PCA)
print("Creating classification visualization...")

# Prepare data for PCA
feature_data = df[numeric_cols].values
scaler = StandardScaler()
scaled_data = scaler.fit_transform(feature_data)

# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# Create scatter plot
plt.figure(figsize=(12, 10))

colors = ['#3498db', '#e74c3c', '#2ecc71']
markers = ['o', 's', '^']

for idx, class_name in enumerate(CLASS_NAMES):
    mask = df['class'] == class_name
    plt.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                c=colors[idx], 
                marker=markers[idx],
                label=class_name,
                alpha=0.7,
                s=100,
                edgecolors='black',
                linewidth=0.5)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12, fontweight='bold')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12, fontweight='bold')
plt.title('PCA Classification Visualization - Lymphoma Types', fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=11, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'classification_visualization.png'), dpi=300, bbox_inches='tight')
print("✓ Classification visualization saved")

# 3. Feature importance by class
print("Creating feature importance visualization...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Feature Distribution by Class', fontsize=16, fontweight='bold')

# Select some key features for visualization
key_features = ['brightness', 'contrast', 'sharpness', 'edge_density']
feature_titles = ['Brightness', 'Contrast', 'Sharpness', 'Edge Density']

for idx, (feature, title) in enumerate(zip(key_features, feature_titles)):
    row = idx // 2
    col = idx % 2
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_data = df[df['class'] == class_name][feature]
        axes[row, col].hist(class_data, bins=20, alpha=0.6, 
                           label=class_name, color=colors[class_idx])
    
    axes[row, col].set_xlabel(title, fontsize=11, fontweight='bold')
    axes[row, col].set_ylabel('Frequency', fontsize=11)
    axes[row, col].set_title(f'{title} Distribution by Class', fontsize=12, fontweight='bold')
    axes[row, col].legend(fontsize=9)
    axes[row, col].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
print("✓ Feature importance visualization saved")

print("\nVisualization complete!")
print(f"Files saved to {OUTPUT_DIR}/")
print("  - correlation_heatmap.png")
print("  - classification_visualization.png") 
print("  - feature_importance.png")
