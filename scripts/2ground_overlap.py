import os
import json
import numpy as np
import cv2
import argparse
from pycocotools import mask as maskUtils
import h5py
import tqdm

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

def get_touched_ground_mask(annotations, keypoints, height, width, radius=20):
    touched_mask = np.zeros((height, width), dtype=np.uint8)

    for pt in keypoints:
        px, py = pt
        x1, x2 = max(0, px - radius), min(width, px + radius + 1)
        y1, y2 = max(0, py - radius), min(height, py + radius + 1)
        foot_patch = np.zeros((height, width), dtype=np.uint8)
        foot_patch[y1:y2, x1:x2] = 1

        best_iou = 0
        best_mask = None

        for ann in annotations:
            # if ann['class_name'] == 'walkway':
            if ann['class_name'] == 'walkway [SEP]':
                binary_mask = maskUtils.decode(ann['segmentation'])
                patch = binary_mask[y1:y2, x1:x2]
                if np.any(patch):
                    iou = compute_iou(foot_patch, binary_mask)
                    if iou > best_iou:
                        best_iou = iou
                        best_mask = binary_mask

        if best_iou > 0 and best_mask is not None:
            touched_mask = np.logical_or(touched_mask, best_mask).astype(np.uint8)

    return touched_mask

def process_frame(ground_h5_file, keypoints_data, mask_output_group, frame_name, crop_offsets):
    frame_group = ground_h5_file[frame_name]
    # frame_group = ground_h5_file[frame_name]
    annotations = json.loads(frame_group['annotations'][()])

    offset = crop_offsets.get(frame_name, {"x1": 0, "y1": 0})
    dx, dy = offset["x1"], offset["y1"]

    foot_points = []
    # Extract left foot (19-21) and right foot (22-24) keypoints
    for indices in [(19, 20, 21), (22, 23, 24)]:
        for idx in indices:
            px, py,conf = keypoints_data[idx]
            foot_points.append((int(px + dx), int(py + dy)))

    # Use get_touched_ground_mask to compute the touched mask
    touched_mask = get_touched_ground_mask(
        annotations=annotations,
        keypoints=foot_points,
        height=frame_group.attrs['img_height'],
        width=frame_group.attrs['img_width']
    )
    # apply morphological operations to clean up the mask
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # touched_mask = cv2.morphologyEx(touched_mask, cv2.MORPH_CLOSE, kernel)
    # touched_mask = cv2.morphologyEx(touched_mask, cv2.MORPH_OPEN, kernel)
    # touched_mask = np.clip(touched_mask, 0, 1)  # Ensure binary mask

    # keep only the largest connected component
    if np.sum(touched_mask) == 0:
        print(f"Warning: No touched ground detected in frame {frame_name}. Skipping.")
        largest_component_mask = np.zeros_like(touched_mask)
    else:
        _, labels, stats, _ = cv2.connectedComponentsWithStats(touched_mask, connectivity=8)

        # Find the label of the largest component (excluding background)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # skip background

        # Create a mask with only the largest component
        largest_component_mask = np.zeros_like(touched_mask)
        largest_component_mask[labels == largest_label] = 255

    mask_output_group.create_dataset(frame_name, data=largest_component_mask, dtype='uint8')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch detect foot-ground overlap using keypoints and segmentation mask.")
    parser.add_argument("--keypoints_h5", required=True, help="Path to HDF5 file containing keypoints.")
    parser.add_argument("--ground_h5", required=True, help="Path to HDF5 file containing ground data.")
    parser.add_argument("--output_h5", required=True, help="Path to output HDF5 file to save binary masks.")
    parser.add_argument("--offset_json", required=True, help="Path to the crop offsets JSON file.")
    args = parser.parse_args()

    # Update to process HDF5 files for keypoints and ground data
    keypoints_h5 = h5py.File(args.keypoints_h5, 'r')
    ground_h5 = h5py.File(args.ground_h5, 'r')
    ground_h5 = ground_h5['results']

    # Compute binary masks for foot-ground overlap using IoU
    # Save binary masks into an HDF5 file
    output_h5 = h5py.File(args.output_h5, 'w')
    binary_masks = output_h5.create_group('binary_masks')

    for frame_name in tqdm.tqdm(keypoints_h5['pose_keypoints']):
        keypoints_data = keypoints_h5['pose_keypoints'][frame_name][:]

        process_frame(
            ground_h5_file=ground_h5,
            keypoints_data=keypoints_data,
            mask_output_group=binary_masks,
            frame_name=frame_name,
            crop_offsets=json.load(open(args.offset_json))
        )

    print(f"\n✅ All binary masks saved to {args.output_h5}")
