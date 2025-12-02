import os
import matplotlib.pyplot as plt
import numpy as np

# Set plot style for better visualization
plt.rcParams["font.family"] = "Arial"
plt.style.use('seaborn-v0_8-darkgrid')

# Path to the dataset
dataset_path = '/home/houhao/workspace/PersonStepMat/materials/SGWalkwayMaterials'

# Get label directories
labels = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

# Count images per label
label_counts = {}
for label in labels:
    label_dir = os.path.join(dataset_path, label)
    images = [f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))]
    label_counts[label] = len(images)

# Sort by count for better visualization
sorted_items = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
sorted_labels = [item[0] for item in sorted_items]
sorted_counts = [item[1] for item in sorted_items]

# Create color gradient based on values
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_labels)))

# Plot with improved styling
fig, ax = plt.subplots(figsize=(14, 7))
bars = ax.bar(range(len(sorted_labels)), sorted_counts, color=colors, 
              edgecolor='black', linewidth=1.2, alpha=0.85)

# Customize axes
ax.set_xlabel('Material Labels', fontsize=30, labelpad=10)
ax.set_ylabel('Number of Images', fontsize=30, labelpad=10)
ax.set_title('Class Distribution of the SGWalkwayMaterials Dataset', 
             fontsize=30, pad=20)

# Set x-axis labels
ax.set_xticks(range(len(sorted_labels)))
ax.set_xticklabels(sorted_labels, rotation=45, ha='right', fontsize=18)
ax.tick_params(axis='y', labelsize=18)
ax.set_ylim(0, max(sorted_counts) * 1.08)
# Add grid for better readability
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax.set_axisbelow(True)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

# Add value labels on top of each bar with better formatting
for i, (bar, count) in enumerate(zip(bars, sorted_counts)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + max(sorted_counts) * 0.01, 
            f'{int(count)}', ha='center', va='bottom', fontsize=14, fontweight='bold')

# Add summary statistics
total_images = sum(sorted_counts)
avg_images = np.mean(sorted_counts)
ax.text(0.98, 0.97, f'Total: {total_images}\nAvg: {avg_images:.1f}\nClasses: {len(sorted_labels)}',
        transform=ax.transAxes, fontsize=14, verticalalignment='top',
        horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', 
        alpha=0.8, edgecolor='gray', linewidth=1.5))

# Adjust layout
plt.tight_layout()

# Save with high quality
plt.savefig("image_count_per_label.png", dpi=300, bbox_inches='tight')
plt.savefig("image_count_per_label.pdf", bbox_inches='tight')
print(f"Saved plots: image_count_per_label.png and image_count_per_label.pdf")
print(f"\nDataset Statistics:")
print(f"  Total images: {total_images}")
print(f"  Number of classes: {len(sorted_labels)}")
print(f"  Average images per class: {avg_images:.1f}")
print(f"  Min images: {min(sorted_counts)} ({sorted_labels[sorted_counts.index(min(sorted_counts))]})")
print(f"  Max images: {max(sorted_counts)} ({sorted_labels[sorted_counts.index(max(sorted_counts))]})")
plt.show()