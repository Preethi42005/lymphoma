import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np

# Set up the figure with subplots for all architectures
fig = plt.figure(figsize=(20, 24))
fig.suptitle('Deep Learning Architecture Diagrams', fontsize=20, fontweight='bold', y=0.995)

# Helper function to draw boxes and arrows
def draw_box(ax, x, y, width, height, text, color='lightblue', fontsize=10):
    box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1", 
                          edgecolor='black', facecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', 
            fontsize=fontsize, fontweight='bold', wrap=True)
    return box

def draw_arrow(ax, x1, y1, x2, y2, text=''):
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', 
                           mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow)
    if text:
        mid_x, mid_y = (x1 + x2)/2, (y1 + y2)/2
        ax.text(mid_x, mid_y + 0.02, text, ha='center', fontsize=8)

def draw_circle(ax, x, y, radius, text, color='lightgreen'):
    circle = Circle((x, y), radius, edgecolor='black', facecolor=color, linewidth=2)
    ax.add_patch(circle)
    ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

# 1. Multi-Modal Fusion Architecture
ax1 = plt.subplot(4, 1, 1)
ax1.set_title('Multi-Modal Fusion Architecture', fontsize=14, fontweight='bold')
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 6)
ax1.axis('off')

# Input modalities
draw_box(ax1, 0.5, 4.5, 1.5, 0.8, 'Image\nInput', 'lightcoral', 9)
draw_box(ax1, 0.5, 2.5, 1.5, 0.8, 'Text\nInput', 'lightgreen', 9)
draw_box(ax1, 0.5, 0.5, 1.5, 0.8, 'Clinical\nData', 'lightyellow', 9)

# Feature extractors
draw_box(ax1, 3, 4.5, 1.5, 0.8, 'CNN\nEncoder', 'lightblue', 9)
draw_box(ax1, 3, 2.5, 1.5, 0.8, 'BERT/\nTransformer', 'lightblue', 9)
draw_box(ax1, 3, 0.5, 1.5, 0.8, 'MLP\nEncoder', 'lightblue', 9)

# Arrows from inputs to encoders
draw_arrow(ax1, 2, 4.9, 3, 4.9)
draw_arrow(ax1, 2, 2.9, 3, 2.9)
draw_arrow(ax1, 2, 0.9, 3, 0.9)

# Feature embeddings
draw_box(ax1, 5.5, 4.5, 1.5, 0.8, 'Image\nEmbedding', 'lavender', 9)
draw_box(ax1, 5.5, 2.5, 1.5, 0.8, 'Text\nEmbedding', 'lavender', 9)
draw_box(ax1, 5.5, 0.5, 1.5, 0.8, 'Clinical\nEmbedding', 'lavender', 9)

# Arrows from encoders to embeddings
draw_arrow(ax1, 4.5, 4.9, 5.5, 4.9)
draw_arrow(ax1, 4.5, 2.9, 5.5, 2.9)
draw_arrow(ax1, 4.5, 0.9, 5.5, 0.9)

# Fusion layer
draw_box(ax1, 8, 2.5, 1.5, 1.5, 'Multi-Modal\nFusion\n(Attention/\nConcatenation)', 'orange', 10)

# Arrows to fusion
draw_arrow(ax1, 7, 4.9, 8, 3.5)
draw_arrow(ax1, 7, 2.9, 8, 2.9)
draw_arrow(ax1, 7, 0.9, 8, 1.5)

# Output
draw_box(ax1, 8, 0.2, 1.5, 0.6, 'Classification\nOutput', 'lightcoral', 9)
draw_arrow(ax1, 8.75, 2.5, 8.75, 0.8)

# 2. Deep Learning Hybrid Model Architecture
ax2 = plt.subplot(4, 1, 2)
ax2.set_title('Deep Learning Hybrid Model Architecture (MobileNetV2 + DenseNet121)', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 12)
ax2.set_ylim(0, 6)
ax2.axis('off')

# Input
draw_box(ax2, 0.5, 2.5, 1.5, 1, 'Input\nImage\n(128x128x3)', 'lightcoral', 10)

# Backbone 1: MobileNetV2
draw_box(ax2, 3, 4.5, 2, 0.6, 'MobileNetV2', 'lightblue', 10)
draw_box(ax2, 3, 3.7, 2, 0.5, 'Low-Scale Features', 'lavender', 9)
draw_box(ax2, 3, 3.0, 2, 0.5, 'Mid-Scale Features', 'lavender', 9)
draw_box(ax2, 3, 2.3, 2, 0.5, 'High-Scale Features', 'lavender', 9)

# Backbone 2: DenseNet121
draw_box(ax2, 3, 1.3, 2, 0.6, 'DenseNet121', 'lightgreen', 10)
draw_box(ax2, 3, 0.5, 2, 0.5, 'Low-Scale Features', 'lavender', 9)
draw_box(ax2, 3, -0.2, 2, 0.5, 'Mid-Scale Features', 'lavender', 9)
draw_box(ax2, 3, -0.9, 2, 0.5, 'High-Scale Features', 'lavender', 9)

# Arrows from input to backbones
draw_arrow(ax2, 2, 3.5, 3, 4.8)
draw_arrow(ax2, 2, 2.5, 3, 1.6)

# Feature fusion
draw_box(ax2, 6, 1.8, 2, 1.5, 'Multi-Scale\nFeature Fusion\n(Concatenation +\nAttention)', 'orange', 10)

# Arrows to fusion
draw_arrow(ax2, 5, 2.5, 6, 2.5)
draw_arrow(ax2, 5, 0.8, 6, 1.8)

# Classification head
draw_box(ax2, 9, 2.5, 1, 0.5, 'Dense\n512', 'lightyellow', 9)
draw_box(ax2, 9, 1.8, 1, 0.5, 'Dropout\n0.5', 'lightgray', 9)
draw_box(ax2, 9, 1.1, 1, 0.5, 'Dense\n256', 'lightyellow', 9)
draw_box(ax2, 9, 0.4, 1, 0.5, 'Dropout\n0.4', 'lightgray', 9)
draw_box(ax2, 9, -0.3, 1, 0.5, 'Dense\n128', 'lightyellow', 9)

# Arrows through classification head
draw_arrow(ax2, 8, 2.5, 9, 2.75)
draw_arrow(ax2, 9.5, 2.5, 9.5, 2.3)
draw_arrow(ax2, 9.5, 1.8, 9.5, 1.6)
draw_arrow(ax2, 9.5, 1.1, 9.5, 0.9)
draw_arrow(ax2, 9.5, 0.4, 9.5, 0.2)

# Output
draw_box(ax2, 10.5, 0.5, 1.2, 0.8, 'Softmax\n(3 Classes)', 'lightcoral', 9)
draw_arrow(ax2, 10, -0.05, 10.5, 0.5)

# 3. MobileNet Architecture
ax3 = plt.subplot(4, 1, 3)
ax3.set_title('MobileNetV2 Architecture', fontsize=14, fontweight='bold')
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 4)
ax3.axis('off')

# Input
draw_box(ax3, 0.3, 1.5, 1, 0.8, 'Input\n(224x224x3)', 'lightcoral', 9)

# Initial convolution
draw_box(ax3, 1.8, 1.5, 1, 0.8, 'Conv\n3x3\ns=2', 'lightblue', 9)

# Bottleneck layers (repeated)
for i in range(3):
    y_pos = 3.2 - i * 1.2
    draw_box(ax3, 3.5, y_pos, 1.2, 0.8, f'Bottleneck\n{i+1}\n1x1, 3x3', 'lightgreen', 8)
    if i < 2:
        draw_arrow(ax3, 4.1, y_pos, 4.1, y_pos - 1.2)

# Connect input to first bottleneck
draw_arrow(ax3, 1.3, 1.9, 1.8, 1.9)
draw_arrow(ax3, 2.8, 1.9, 3.5, 3.2)

# Final layers
draw_box(ax3, 5.2, 1.5, 1, 0.8, 'Conv\n1x1', 'lightblue', 9)
draw_box(ax3, 6.7, 1.5, 1, 0.8, 'AvgPool\n7x7', 'lavender', 9)
draw_box(ax3, 8.2, 1.5, 1, 0.8, 'FC\nLayer', 'lightyellow', 9)

# Connect final layers
draw_arrow(ax3, 4.7, 0.8, 5.2, 1.9)
draw_arrow(ax3, 6.2, 1.9, 6.7, 1.9)
draw_arrow(ax3, 7.7, 1.9, 8.2, 1.9)

# Output
draw_box(ax3, 9.5, 1.5, 0.8, 0.8, 'Softmax', 'lightcoral', 9)
draw_arrow(ax3, 9.2, 1.9, 9.5, 1.9)

# Add depthwise separable convolution note
ax3.text(5, 0.3, 'Key Feature: Depthwise Separable Convolutions', 
         ha='center', fontsize=10, style='italic', 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 4. CNN-LSTM Architecture
ax4 = plt.subplot(4, 1, 4)
ax4.set_title('CNN-LSTM Architecture for Sequence/Video Analysis', fontsize=14, fontweight='bold')
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 5)
ax4.axis('off')

# Input sequence
for i in range(3):
    draw_box(ax4, 0.3, 3.5 - i * 1.2, 1, 0.8, f'Frame\n{i+1}', 'lightcoral', 9)

# CNN feature extractors
for i in range(3):
    y_pos = 3.5 - i * 1.2
    draw_box(ax4, 1.8, y_pos, 1, 0.8, 'CNN\nFeatures', 'lightblue', 9)
    draw_arrow(ax4, 1.3, y_pos + 0.4, 1.8, y_pos + 0.4)

# Flatten layers
for i in range(3):
    y_pos = 3.5 - i * 1.2
    draw_box(ax4, 3.3, y_pos, 1, 0.8, 'Flatten', 'lavender', 9)
    draw_arrow(ax4, 2.8, y_pos + 0.4, 3.3, y_pos + 0.4)

# Sequence formation
draw_box(ax4, 4.8, 2.5, 1.2, 1.5, 'Sequence\nFormation', 'lightyellow', 10)

# Arrows to sequence
draw_arrow(ax4, 4.3, 3.9, 4.8, 3.5)
draw_arrow(ax4, 4.3, 2.7, 4.8, 2.7)
draw_arrow(ax4, 4.3, 1.5, 4.8, 1.5)

# LSTM layers
draw_box(ax4, 6.5, 3.2, 1.2, 0.8, 'LSTM\nLayer 1', 'lightgreen', 9)
draw_box(ax4, 6.5, 2.2, 1.2, 0.8, 'LSTM\nLayer 2', 'lightgreen', 9)

# Arrows to LSTM
draw_arrow(ax4, 6, 3.2, 6.5, 3.6)
draw_arrow(ax4, 6, 2.5, 6.5, 2.6)
draw_arrow(ax4, 7.1, 3.2, 7.1, 3.0)

# Dense layers
draw_box(ax4, 8.2, 2.5, 1, 0.8, 'Dense\n128', 'lightyellow', 9)
draw_box(ax4, 8.2, 1.5, 1, 0.6, 'Dropout', 'lightgray', 9)

# Arrows to dense
draw_arrow(ax4, 7.7, 2.6, 8.2, 2.9)
draw_arrow(ax4, 8.7, 2.5, 8.7, 2.1)

# Output
draw_box(ax4, 9.5, 1.5, 0.8, 0.8, 'Output', 'lightcoral', 9)
draw_arrow(ax4, 9.2, 1.8, 9.5, 1.9)

# Add temporal processing note
ax4.text(5, 0.3, 'Key Feature: Temporal Sequence Processing', 
         ha='center', fontsize=10, style='italic',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('architecture_diagrams.png', dpi=300, bbox_inches='tight')
print("✓ Architecture diagrams saved to 'architecture_diagrams.png'")

# Create individual diagrams for better clarity
# 1. Multi-Modal Fusion
fig1, ax1 = plt.subplots(figsize=(12, 8))
ax1.set_title('Multi-Modal Fusion Architecture', fontsize=16, fontweight='bold')
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 6)
ax1.axis('off')

# Input modalities
draw_box(ax1, 0.5, 4.5, 1.5, 0.8, 'Image\nInput', 'lightcoral', 11)
draw_box(ax1, 0.5, 2.5, 1.5, 0.8, 'Text\nInput', 'lightgreen', 11)
draw_box(ax1, 0.5, 0.5, 1.5, 0.8, 'Clinical\nData', 'lightyellow', 11)

# Feature extractors
draw_box(ax1, 3, 4.5, 1.5, 0.8, 'CNN\nEncoder', 'lightblue', 11)
draw_box(ax1, 3, 2.5, 1.5, 0.8, 'BERT/\nTransformer', 'lightblue', 11)
draw_box(ax1, 3, 0.5, 1.5, 0.8, 'MLP\nEncoder', 'lightblue', 11)

# Arrows from inputs to encoders
draw_arrow(ax1, 2, 4.9, 3, 4.9)
draw_arrow(ax1, 2, 2.9, 3, 2.9)
draw_arrow(ax1, 2, 0.9, 3, 0.9)

# Feature embeddings
draw_box(ax1, 5.5, 4.5, 1.5, 0.8, 'Image\nEmbedding', 'lavender', 11)
draw_box(ax1, 5.5, 2.5, 1.5, 0.8, 'Text\nEmbedding', 'lavender', 11)
draw_box(ax1, 5.5, 0.5, 1.5, 0.8, 'Clinical\nEmbedding', 'lavender', 11)

# Arrows from encoders to embeddings
draw_arrow(ax1, 4.5, 4.9, 5.5, 4.9)
draw_arrow(ax1, 4.5, 2.9, 5.5, 2.9)
draw_arrow(ax1, 4.5, 0.9, 5.5, 0.9)

# Fusion layer
draw_box(ax1, 8, 2.5, 1.5, 1.5, 'Multi-Modal\nFusion\n(Attention/\nConcatenation)', 'orange', 11)

# Arrows to fusion
draw_arrow(ax1, 7, 4.9, 8, 3.5)
draw_arrow(ax1, 7, 2.9, 8, 2.9)
draw_arrow(ax1, 7, 0.9, 8, 1.5)

# Output
draw_box(ax1, 8, 0.2, 1.5, 0.6, 'Classification\nOutput', 'lightcoral', 11)
draw_arrow(ax1, 8.75, 2.5, 8.75, 0.8)

plt.tight_layout()
plt.savefig('multimodal_fusion_architecture.png', dpi=300, bbox_inches='tight')
print("✓ Multi-modal fusion diagram saved")

# 2. Hybrid Model
fig2, ax2 = plt.subplots(figsize=(14, 8))
ax2.set_title('Deep Learning Hybrid Model Architecture (MobileNetV2 + DenseNet121)', fontsize=16, fontweight='bold')
ax2.set_xlim(0, 14)
ax2.set_ylim(0, 7)
ax2.axis('off')

# Input
draw_box(ax2, 0.5, 3, 1.5, 1, 'Input\nImage\n(128x128x3)', 'lightcoral', 11)

# Backbone 1: MobileNetV2
draw_box(ax2, 3, 5.5, 2, 0.6, 'MobileNetV2', 'lightblue', 11)
draw_box(ax2, 3, 4.7, 2, 0.5, 'Low-Scale Features', 'lavender', 10)
draw_box(ax2, 3, 4.0, 2, 0.5, 'Mid-Scale Features', 'lavender', 10)
draw_box(ax2, 3, 3.3, 2, 0.5, 'High-Scale Features', 'lavender', 10)

# Backbone 2: DenseNet121
draw_box(ax2, 3, 2.3, 2, 0.6, 'DenseNet121', 'lightgreen', 11)
draw_box(ax2, 3, 1.5, 2, 0.5, 'Low-Scale Features', 'lavender', 10)
draw_box(ax2, 3, 0.8, 2, 0.5, 'Mid-Scale Features', 'lavender', 10)
draw_box(ax2, 3, 0.1, 2, 0.5, 'High-Scale Features', 'lavender', 10)

# Arrows from input to backbones
draw_arrow(ax2, 2, 4, 3, 5.8)
draw_arrow(ax2, 2, 3, 3, 2.6)

# Feature fusion
draw_box(ax2, 6, 2.8, 2, 1.5, 'Multi-Scale\nFeature Fusion\n(Concatenation +\nAttention)', 'orange', 11)

# Arrows to fusion
draw_arrow(ax2, 5, 3.5, 6, 3.5)
draw_arrow(ax2, 5, 1.2, 6, 2.8)

# Classification head
draw_box(ax2, 9, 3.5, 1, 0.5, 'Dense\n512', 'lightyellow', 10)
draw_box(ax2, 9, 2.8, 1, 0.5, 'Dropout\n0.5', 'lightgray', 10)
draw_box(ax2, 9, 2.1, 1, 0.5, 'Dense\n256', 'lightyellow', 10)
draw_box(ax2, 9, 1.4, 1, 0.5, 'Dropout\n0.4', 'lightgray', 10)
draw_box(ax2, 9, 0.7, 1, 0.5, 'Dense\n128', 'lightyellow', 10)

# Arrows through classification head
draw_arrow(ax2, 8, 3.5, 9, 3.75)
draw_arrow(ax2, 9.5, 3.5, 9.5, 3.3)
draw_arrow(ax2, 9.5, 2.8, 9.5, 2.6)
draw_arrow(ax2, 9.5, 2.1, 9.5, 1.9)
draw_arrow(ax2, 9.5, 1.4, 9.5, 1.2)

# Output
draw_box(ax2, 10.5, 0.5, 1.2, 0.8, 'Softmax\n(3 Classes)', 'lightcoral', 10)
draw_arrow(ax2, 10, 0.7, 10.5, 0.9)

# Add legend
ax2.text(7, 0.2, 'Blue: MobileNetV2 | Green: DenseNet121 | Orange: Fusion Layer', 
         ha='center', fontsize=9, style='italic',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('hybrid_model_architecture.png', dpi=300, bbox_inches='tight')
print("✓ Hybrid model diagram saved")

# 3. MobileNet
fig3, ax3 = plt.subplots(figsize=(12, 6))
ax3.set_title('MobileNetV2 Architecture', fontsize=16, fontweight='bold')
ax3.set_xlim(0, 12)
ax3.set_ylim(0, 5)
ax3.axis('off')

# Input
draw_box(ax3, 0.3, 2, 1, 0.8, 'Input\n(224x224x3)', 'lightcoral', 11)

# Initial convolution
draw_box(ax3, 1.8, 2, 1, 0.8, 'Conv\n3x3\ns=2', 'lightblue', 11)

# Bottleneck layers (repeated)
for i in range(4):
    y_pos = 4.0 - i * 1.0
    draw_box(ax3, 3.5, y_pos, 1.2, 0.8, f'Bottleneck\n{i+1}\n1x1, 3x3', 'lightgreen', 10)
    if i < 3:
        draw_arrow(ax3, 4.1, y_pos, 4.1, y_pos - 1.0)

# Connect input to first bottleneck
draw_arrow(ax3, 1.3, 2.4, 1.8, 2.4)
draw_arrow(ax3, 2.8, 2.4, 3.5, 4.0)

# Final layers
draw_box(ax3, 5.2, 2, 1, 0.8, 'Conv\n1x1', 'lightblue', 11)
draw_box(ax3, 6.7, 2, 1, 0.8, 'AvgPool\n7x7', 'lavender', 11)
draw_box(ax3, 8.2, 2, 1, 0.8, 'FC\nLayer', 'lightyellow', 11)

# Connect final layers
draw_arrow(ax3, 4.7, 1.2, 5.2, 2.4)
draw_arrow(ax3, 6.2, 2.4, 6.7, 2.4)
draw_arrow(ax3, 7.7, 2.4, 8.2, 2.4)

# Output
draw_box(ax3, 9.5, 2, 0.8, 0.8, 'Softmax', 'lightcoral', 11)
draw_arrow(ax3, 9.2, 2.4, 9.5, 2.4)

# Add depthwise separable convolution note
ax3.text(6, 0.3, 'Key Feature: Depthwise Separable Convolutions for Efficiency', 
         ha='center', fontsize=11, style='italic', 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('mobilenet_architecture.png', dpi=300, bbox_inches='tight')
print("✓ MobileNet architecture diagram saved")

# 4. CNN-LSTM
fig4, ax4 = plt.subplots(figsize=(12, 8))
ax4.set_title('CNN-LSTM Architecture for Sequence/Video Analysis', fontsize=16, fontweight='bold')
ax4.set_xlim(0, 12)
ax4.set_ylim(0, 6)
ax4.axis('off')

# Input sequence
for i in range(4):
    draw_box(ax4, 0.3, 4.5 - i * 1.1, 1, 0.8, f'Frame\n{i+1}', 'lightcoral', 10)

# CNN feature extractors
for i in range(4):
    y_pos = 4.5 - i * 1.1
    draw_box(ax4, 1.8, y_pos, 1, 0.8, 'CNN\nFeatures', 'lightblue', 10)
    draw_arrow(ax4, 1.3, y_pos + 0.4, 1.8, y_pos + 0.4)

# Flatten layers
for i in range(4):
    y_pos = 4.5 - i * 1.1
    draw_box(ax4, 3.3, y_pos, 1, 0.8, 'Flatten', 'lavender', 10)
    draw_arrow(ax4, 2.8, y_pos + 0.4, 3.3, y_pos + 0.4)

# Sequence formation
draw_box(ax4, 4.8, 2.8, 1.2, 1.5, 'Sequence\nFormation', 'lightyellow', 11)

# Arrows to sequence
draw_arrow(ax4, 4.3, 4.9, 4.8, 3.8)
draw_arrow(ax4, 4.3, 3.8, 4.8, 3.1)
draw_arrow(ax4, 4.3, 2.7, 4.8, 2.8)
draw_arrow(ax4, 4.3, 1.6, 4.8, 1.8)

# LSTM layers
draw_box(ax4, 6.5, 3.8, 1.2, 0.8, 'LSTM\nLayer 1', 'lightgreen', 11)
draw_box(ax4, 6.5, 2.8, 1.2, 0.8, 'LSTM\nLayer 2', 'lightgreen', 11)

# Arrows to LSTM
draw_arrow(ax4, 6, 3.5, 6.5, 4.2)
draw_arrow(ax4, 6, 2.8, 6.5, 3.2)
draw_arrow(ax4, 7.1, 3.8, 7.1, 3.6)

# Dense layers
draw_box(ax4, 8.2, 3.0, 1, 0.8, 'Dense\n128', 'lightyellow', 11)
draw_box(ax4, 8.2, 2.0, 1, 0.6, 'Dropout', 'lightgray', 11)

# Arrows to dense
draw_arrow(ax4, 7.7, 3.2, 8.2, 3.4)
draw_arrow(ax4, 8.7, 3.0, 8.7, 2.6)

# Output
draw_box(ax4, 9.5, 2.0, 0.8, 0.8, 'Output', 'lightcoral', 11)
draw_arrow(ax4, 9.2, 2.3, 9.5, 2.4)

# Add temporal processing note
ax4.text(6, 0.3, 'Key Feature: Temporal Sequence Processing with Memory', 
         ha='center', fontsize=11, style='italic',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('cnn_lstm_architecture.png', dpi=300, bbox_inches='tight')
print("✓ CNN-LSTM architecture diagram saved")

print("\nAll architecture diagrams generated successfully!")
print("Generated files:")
print("  - architecture_diagrams.png (combined)")
print("  - multimodal_fusion_architecture.png")
print("  - hybrid_model_architecture.png")
print("  - mobilenet_architecture.png")
print("  - cnn_lstm_architecture.png")
