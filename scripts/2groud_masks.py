import os
import json
import numpy as np
from PIL import Image

from pycocotools import mask as maskUtils

# Paths
json_dir = '/home/houhao/workspace/PersonStepMat/dataset/pilot2/walkway_width'
output_dir = '/home/houhao/workspace/PersonStepMat/dataset/pilot2/walkway_masks'
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(json_dir):
    if not fname.endswith('.json'):
        continue
    with open(os.path.join(json_dir, fname), 'r') as f:
        data = json.load(f)

    # New format: 'annotations' is a list, each with 'class_name', 'segmentation', etc.
    mask = np.zeros((data['img_height'], data['img_width']), dtype=np.uint8)
    for ann in data.get('annotations', []):
        if ann.get('class_name', '').lower() == 'walkway':
            seg = ann['segmentation']
            rle = {
                'counts': seg['counts'],
                'size': seg['size']
            }
            m = maskUtils.decode(rle)
            mask = np.maximum(mask, m)  # Combine masks if multiple

    mask = (mask > 0).astype(np.uint8) * 255
    # Extract the frame index from the filename, e.g., 'results_frame_00000.json'
    base = os.path.splitext(fname)[0]  # 'results_frame_00000'
    idx_str = base.split('_')[-1]      # '00000'
    idx = int(idx_str)
    out_path = os.path.join(output_dir, f'frame_{idx:05d}_ground_masks.png')
    Image.fromarray(mask).save(out_path)