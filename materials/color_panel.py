import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
import matplotlib.patches as mpatches
import numpy as np

# Label definitions from SGWalkwayMaterials.py
labels_info = [
    {'trainId': 0, 'name': 'AcrylicPolymer', 'color': (31, 119, 180)},
    {'trainId': 1, 'name': 'Asphalt', 'color': (44, 160, 44)},
    {'trainId': 2, 'name': 'BicycleCoat', 'color': (214, 39, 40)},
    {'trainId': 3, 'name': 'Bricks', 'color': (255, 127, 14)},
    {'trainId': 4, 'name': 'Brushed_Concrete_New', 'color': (148, 103, 189)},
    {'trainId': 5, 'name': 'Brushed_Concrete_Old', 'color': (140, 86, 75)},
    {'trainId': 6, 'name': 'Ceramic_Porcelain_Tiles', 'color': (227, 119, 194)},
    {'trainId': 7, 'name': 'ConcretePavers', 'color': (127, 127, 127)},
    {'trainId': 8, 'name': 'Dirt', 'color': (188, 189, 34)},
    {'trainId': 9, 'name': 'Exposed_Aggregate_Concrete', 'color': (23, 190, 207)},
    {'trainId': 10, 'name': 'Granite', 'color': (174, 199, 232)},
    {'trainId': 11, 'name': 'Grass', 'color': (152, 223, 138)},
    {'trainId': 12, 'name': 'Metal', 'color': (255, 187, 120)},
    {'trainId': 13, 'name': 'Rubber', 'color': (197, 176, 213)},
    # {'trainId': 14, 'name': 'Unknown', 'color': (196, 156, 148)},
]

labels_info1 = [
    {'trainId': 0, 'name': 'concrete', 'color': (255, 127, 14)},
    {'trainId': 1, 'name': 'bricks', 'color': (43, 160, 43)},
    {'trainId': 2, 'name': 'granite', 'color': (31, 119, 179)},
    {'trainId': 3, 'name': 'asphalt', 'color': (153, 153, 153)},
    {'trainId': 4, 'name': 'mixed', 'color': (214, 39, 40)},
    {'trainId': 5, 'name': 'road', 'color': (54, 54, 54)},
    {'trainId': 6, 'name': 'background', 'color': (0, 0, 0)},
    {'trainId': 7, 'name': 'granite block-stone', 'color': (138, 0, 138)},
    {'trainId': 8, 'name': 'hexagonal', 'color': (240, 110, 170)},
    {'trainId': 9, 'name': 'cobblestone', 'color': (139, 109, 48)},
]

def create_color_legend_vertical_dual(save_path='color_legend_dual.png'):
    """
    Create a vertical color legend showing SGWalkwayMaterials and CitySurfaces datasets stacked vertically
    
    Args:
        save_path: Path to save the legend image
    """
    # Calculate total height needed
    total_items = len(labels_info) + len(labels_info1) + 2  # +2 for spacing between sections
    
    # fig, ax = plt.subplots(figsize=(15, max(12, total_items * 0.95)))
    fig, ax = plt.subplots(figsize=(15, 15))
    
    current_y = total_items
    
    # SGWalkwayMaterials (top section)
    colors_norm = [tuple(c/255.0 for c in label['color']) for label in labels_info]
    names = [f"{label['trainId']}: {label['name']}" for label in labels_info]
    
    for i, (color, name) in enumerate(zip(colors_norm, names)):
        current_y -= 1
        rect = mpatches.Rectangle((0, current_y), 1.0, 0.8, facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(1.2, current_y + 0.4, name, va='center', fontsize=34)
    
    # Add section title for SGWalkwayMaterials
    ax.text(0, current_y + len(labels_info), 'SGWalkwayMaterials (Proposed)', 
            fontsize=34, va='bottom')
    
    # Add spacing between sections
    current_y -= 1
    
    # CitySurfaces (bottom section)
    colors_norm1 = [tuple(c/255.0 for c in label['color']) for label in labels_info1]
    names1 = [f"{label['trainId']}: {label['name']}" for label in labels_info1]
    
    # Add section title for CitySurfaces
    ax.text(0, current_y, 'CitySurfaces', 
            fontsize=34, va='bottom')
    
    for i, (color, name) in enumerate(zip(colors_norm1, names1)):
        current_y -= 1
        rect = mpatches.Rectangle((0, current_y), 1.0, 0.8, facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(1.2, current_y + 0.4, name, va='center', fontsize=34)
    
    ax.set_xlim(0, 7)
    ax.set_ylim(0, total_items)
    ax.axis('off')
    plt.title('Material Classification Color Legends', fontsize=34, pad=50, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Dual color legend saved to: {save_path}")
    plt.close()


def create_color_legend_single(labels, title, save_path='color_legend.png'):
    """
    Create a vertical color legend for a single dataset
    
    Args:
        labels: List of label dictionaries
        title: Title for the legend
        save_path: Path to save the legend image
    """
    # Normalize colors from 0-255 to 0-1 for matplotlib
    colors_norm = [tuple(c/255.0 for c in label['color']) for label in labels]
    names = [f"{label['trainId']}: {label['name']}" for label in labels]
    
    fig, ax = plt.subplots(figsize=(7, max(8, len(labels) * 0.6)))
    
    # Create color boxes
    for i, (color, name) in enumerate(zip(colors_norm, names)):
        y_pos = len(labels) - i - 1
        rect = mpatches.Rectangle((0, y_pos), 1.0, 0.8, facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        # Add text label
        ax.text(1.2, y_pos + 0.4, name, va='center', fontsize=14)
    
    ax.set_xlim(0, 6)
    ax.set_ylim(0, len(labels))
    ax.axis('off')
    plt.title(title, fontsize=18, pad=15, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Color legend saved to: {save_path}")
    plt.close()


def create_color_grid(save_path='color_grid.png'):
    """
    Create a grid-style color legend with larger color swatches
    """
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, label in enumerate(labels_info):
        color_norm = tuple(c/255.0 for c in label['color'])
        axes[i].add_patch(mpatches.Rectangle((0, 0), 1, 1, facecolor=color_norm))
        axes[i].set_xlim(0, 1)
        axes[i].set_ylim(0, 1)
        axes[i].axis('off')
        axes[i].set_title(f"{label['trainId']}: {label['name']}", fontsize=30, pad=10)
    
    plt.suptitle('SGWalkwayMaterials Material Classes Color Legend', fontsize=30, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Color grid saved to: {save_path}")
    plt.close()


def print_color_table():
    """
    Print a text table of the color mappings
    """
    print("\n" + "="*70)
    print("SGWalkwayMaterials Dataset - Color to Label Mapping")
    print("="*70)
    print(f"{'Train ID':<10} {'Label Name':<25} {'RGB Color':<20}")
    print("-"*70)
    for label in labels_info:
        rgb_str = f"({label['color'][0]}, {label['color'][1]}, {label['color'][2]})"
        print(f"{label['trainId']:<10} {label['name']:<25} {rgb_str:<20}")
    print("="*70 + "\n")


if __name__ == '__main__':
    # Print text table
    print_color_table()
    
    # Generate dual vertical color legend (both datasets side by side)
    create_color_legend_vertical_dual('color_legend_dual_vertical.png')
    
    # Generate individual vertical legends
    create_color_legend_single(labels_info, 'SGWalkwayMaterials Classes Color Legend', 
                              'color_legend_sgwalkway.png')
    create_color_legend_single(labels_info1, 'CitySurfaces Classes Color Legend', 
                              'color_legend_citysurfaces.png')
    
    # Generate grid for SGWalkwayMaterials
    create_color_grid('color_legend_grid.png')
    
    print("\nAll color legends generated successfully!")
