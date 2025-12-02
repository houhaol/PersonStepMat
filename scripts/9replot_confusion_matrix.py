import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
import seaborn as sns

# Define class names
classes = [
    'AcrylicPolymer',
    'Asphalt',
    'BicycleCoat',
    'Bricks',
    'Brushed_Concrete_New',
    'Brushed_Concrete_Old',
    'Ceramic_Porcelain_Tiles',
    'ConcretePavers',
    'Dirt',
    'Exposed_Aggregate_Concrete',
    'Granite',
    'Grass',
    'Metal',
    'Rubber'
]

# Confusion matrix data (in percentages)
# Rows: True labels, Columns: Predicted labels
confusion_matrix = np.array([
    [85.7, 0.0, 7.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.1, 0.0, 0.0, 0.0],  # AcrylicPolymer
    [0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Asphalt
    [0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # BicycleCoat
    [0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Bricks
    [8.3, 0.0, 0.0, 0.0, 83.3, 8.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Brushed_Concrete_New
    [0.0, 0.0, 0.0, 0.0, 0.0, 94.7, 0.0, 0.0, 2.6, 2.6, 0.0, 0.0, 0.0, 0.0],  # Brushed_Concrete_Old
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Ceramic_Porcelain_Tiles
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # ConcretePavers
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Dirt
    [9.1, 0.0, 0.0, 0.0, 0.0, 4.5, 0.0, 0.0, 0.0, 86.4, 0.0, 0.0, 0.0, 0.0],  # Exposed_Aggregate_Concrete
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0],  # Granite
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0],  # Grass
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0],  # Metal
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0]   # Rubber
])

# Create figure
fig, ax = plt.subplots(figsize=(14, 12))

# Create heatmap with colorbar aligned to heatmap height
sns.heatmap(confusion_matrix, 
            annot=True, 
            fmt='.1f', 
            cmap='Blues', 
            xticklabels=classes,
            yticklabels=classes,
            cbar_kws={'label': 'Percentage', 'shrink': 0.9},
            vmin=0,
            vmax=100,
            square=True,
            linewidths=0.5,
            linecolor='lightgray',
            annot_kws={'size': 18},
            ax=ax)

# Increase colorbar label font size
cbar = ax.collections[0].colorbar
# cbar.set_label('Percentage', size=32, fontweight='bold')
cbar.set_label('Percentage', size=32)
cbar.ax.tick_params(labelsize=16)

plt.title('Confusion Matrix (%)', fontsize=36, pad=20)
plt.xlabel('Predicted label', fontsize=36)
plt.ylabel('True label', fontsize=36)
# plt.title('Confusion Matrix (%)', fontsize=36, fontweight='bold', pad=20)
# plt.xlabel('Predicted label', fontsize=36, fontweight='bold')
# plt.ylabel('True label', fontsize=36, fontweight='bold')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right', fontsize=24)
plt.yticks(rotation=0, fontsize=24)

plt.tight_layout()

# Save the figure
output_path = '/home/houhao/workspace/PersonStepMat/confusion_matrix_replot.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Confusion matrix saved to: {output_path}")

# Also save as PDF
output_path_pdf = '/home/houhao/workspace/PersonStepMat/confusion_matrix_replot.pdf'
plt.savefig(output_path_pdf, bbox_inches='tight')
print(f"Confusion matrix saved to: {output_path_pdf}")

plt.show()

# Print statistics
print("\n=== Classification Statistics ===")
print(f"Number of classes: {len(classes)}")
print(f"\nPer-class accuracy:")
for i, class_name in enumerate(classes):
    accuracy = confusion_matrix[i, i]
    print(f"  {class_name}: {accuracy:.1f}%")

overall_accuracy = np.mean(np.diag(confusion_matrix))
print(f"\nOverall accuracy: {overall_accuracy:.1f}%")

# Calculate confusion details
print("\n=== Confusion Details ===")
for i, true_class in enumerate(classes):
    for j, pred_class in enumerate(classes):
        if i != j and confusion_matrix[i, j] > 0:
            print(f"  {true_class} → {pred_class}: {confusion_matrix[i, j]:.1f}%")
