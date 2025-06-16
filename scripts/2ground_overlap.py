import os
import json
import numpy as np
import cv2
import argparse
from pycocotools import mask as maskUtils
import h5py

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
            if ann['class_name'] == 'ground':
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
    results_group = ground_h5_file['results']
    frame_group = results_group[frame_name]
    annotations = json.loads(frame_group['annotations'][()])

    offset = crop_offsets.get(frame_name, {"x1": 0, "y1": 0})
    dx, dy = offset["x1"], offset["y1"]

    foot_points = []
    # Extract left foot (19-21) and right foot (22-24) keypoints
    for indices in [(19, 20, 21), (22, 23, 24)]:
        for idx in indices:
            px, py = keypoints_data[idx]
            foot_points.append((int(px + dx), int(py + dy)))

    # Use get_touched_ground_mask to compute the touched mask
    touched_mask = get_touched_ground_mask(
        annotations=annotations,
        keypoints=foot_points,
        height=frame_group.attrs['img_height'],
        width=frame_group.attrs['img_width']
    )

    mask_output_group.create_dataset(frame_name, data=touched_mask, dtype='uint8')

def batch_process(keypoints_h5_path, ground_h5_path, output_h5_path, offset_json):
    with open(offset_json, "r") as f:
        crop_offsets = json.load(f)

    with h5py.File(keypoints_h5_path, 'r') as keypoints_h5, \
         h5py.File(ground_h5_path, 'r') as ground_h5, \
         h5py.File(output_h5_path, 'w') as output_h5:

        keypoints_group = keypoints_h5['pose_keypoints']
        mask_output_group = output_h5.create_group('binary_masks')
        
        for frame_name in keypoints_group:
            keypoints_data = keypoints_group[frame_name][:]
            print(f"\n📷 Processing frame: {frame_name}")

            process_frame(
                ground_h5_file=ground_h5,
                keypoints_data=keypoints_data,
                mask_output_group=mask_output_group,
                frame_name=frame_name,
                crop_offsets=crop_offsets
            )

    print(f"\n✅ All binary masks saved to {output_h5_path}")

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

    # Compute binary masks for foot-ground overlap using IoU
    # Save binary masks into an HDF5 file
    output_h5 = h5py.File(args.output_h5, 'w')
    binary_masks = output_h5.create_group('binary_masks')

    for frame_name in keypoints_h5['pose_keypoints']:
        keypoints_data = keypoints_h5['pose_keypoints'][frame_name][:]
        print(f"\n📷 Processing frame: {frame_name}")

        process_frame(
            ground_h5_file=ground_h5,
            keypoints_data=keypoints_data,
            mask_output_group=binary_masks,
            frame_name=frame_name,
            crop_offsets=json.load(open(args.offset_json))
        )

    print(f"\n✅ All binary masks saved to {args.output_h5}")
