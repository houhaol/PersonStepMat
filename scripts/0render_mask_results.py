import h5py
import numpy as np
import cv2
import os

# Paths to your HDF5 files
frames_h5_path = "/home/houhao/workspace/PersonStepMat/dataset/pilot10/frames.h5"
masks_h5_path = "/home/houhao/workspace/PersonStepMat/dataset/pilot10/masks.h5"
output_dir = "/home/houhao/workspace/PersonStepMat/dataset/pilot10/track_verify"
os.makedirs(output_dir, exist_ok=True)

# Set interval for processing frames
interval = 900  # Process every 5th frame

with h5py.File(frames_h5_path, 'r') as frames_h5, h5py.File(masks_h5_path, 'r') as masks_h5:
    frames = frames_h5['frames']
    masks = masks_h5['masks']
    # Process frames at the specified interval
    for i, key in enumerate(frames.keys()):
        if i % interval != 0:
            continue

        frame = frames[key][:]
        mask = masks[key]['mask_0'][:]

        # Ensure mask is binary and 3-channel for overlay
        mask = (mask > 0).astype(np.uint8) * 255
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        red_mask = np.zeros_like(frame)
        red_mask[:, :, 2] = mask  # Red channel

        overlay = cv2.addWeighted(frame, 0.7, red_mask, 0.3, 0)
        out_path = os.path.join(output_dir, f"{key}.png")
        cv2.imwrite(out_path, overlay)
        print(f"Saved overlay: {out_path}")