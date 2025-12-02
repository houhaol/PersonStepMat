import re
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
import numpy as np

# Read the log file
log_file = 'mat_linear_classify_log.txt'

with open(log_file, 'r') as f:
    content = f.read()

# Parse the data
ratios = []
accuracies = []

# Find all ratio patterns
ratio_pattern = r'splitting into support/val by ([\d.]+)\.\.\.'
accuracy_pattern = r'Validation Accuracy: ([\d.]+)%'

# Extract all ratios and accuracies
ratio_matches = re.finditer(ratio_pattern, content)
accuracy_matches = re.finditer(accuracy_pattern, content)

# Convert to lists
for ratio_match, accuracy_match in zip(re.finditer(ratio_pattern, content), 
                                        re.finditer(accuracy_pattern, content)):
    ratio = float(ratio_match.group(1))
    accuracy = float(accuracy_match.group(1))
    ratios.append(ratio)
    accuracies.append(accuracy)

print(f"Total data points: {len(ratios)}")

# Group by ratio and calculate statistics
from collections import defaultdict
ratio_groups = defaultdict(list)
for ratio, acc in zip(ratios, accuracies):
    ratio_groups[ratio].append(acc)

# Calculate mean and std for each ratio
unique_ratios = sorted(ratio_groups.keys())
mean_accuracies = [np.mean(ratio_groups[r]) for r in unique_ratios]
std_accuracies = [np.std(ratio_groups[r]) for r in unique_ratios]
min_accuracies = [np.min(ratio_groups[r]) for r in unique_ratios]
max_accuracies = [np.max(ratio_groups[r]) for r in unique_ratios]

# Convert validation ratio to training percentage
# If val_ratio = 0.1, then train_ratio = 1 - 0.1 = 0.9 = 90%
training_percentages = [(1 - r) * 100 for r in unique_ratios]

print(f"\nUnique ratios: {unique_ratios}")
print(f"Training percentages: {training_percentages}")
print(f"Number of runs per ratio: {[len(ratio_groups[r]) for r in unique_ratios]}")

# Create the plot
plt.figure(figsize=(12, 7))

# Plot individual points
for i, ratio in enumerate(unique_ratios):
    accs = ratio_groups[ratio]
    train_pct = training_percentages[i]
    plt.scatter([train_pct] * len(accs), accs, alpha=0.3, s=50, color='blue')

# Plot mean line with error bars
plt.errorbar(training_percentages, mean_accuracies, yerr=std_accuracies, 
             fmt='o-', linewidth=2, markersize=8, 
             color='red', capsize=5, capthick=2,
             label='Mean ± Std')

# Fill between min and max
plt.fill_between(training_percentages, min_accuracies, max_accuracies, 
                 alpha=0.2, color='gray', label='Min-Max Range')

plt.xlabel('Training Data (%)', fontsize=24)
plt.ylabel('Validation Accuracy (%)', fontsize=24)
# plt.xlabel('Training Data (%)', fontsize=24, fontweight='bold')
# plt.ylabel('Validation Accuracy (%)', fontsize=24, fontweight='bold')
plt.title('Linear Classifier Validation Accuracy vs Training Data\n', 
          fontsize=24)
        #   fontsize=24, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)

# Set y-axis limits with some padding
plt.ylim([75, 102])

# Add text box with statistics
best_idx = mean_accuracies.index(max(mean_accuracies))
worst_idx = mean_accuracies.index(min(mean_accuracies))
textstr = f'Best: {max(mean_accuracies):.2f}% at {training_percentages[best_idx]:.0f}% training\n'
textstr += f'Worst: {min(mean_accuracies):.2f}% at {training_percentages[worst_idx]:.0f}% training'
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, 
         fontsize=18, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save the figure
output_file = 'accuracy_vs_ratio.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✅ Plot saved to: {output_file}")

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
for i, ratio in enumerate(unique_ratios):
    train_pct = training_percentages[i]
    print(f"Training {train_pct:.0f}% (val {ratio:.1f}): Mean={mean_accuracies[i]:.2f}%, "
          f"Std={std_accuracies[i]:.2f}%, "
          f"Min={min_accuracies[i]:.2f}%, "
          f"Max={max_accuracies[i]:.2f}%")

plt.show()
