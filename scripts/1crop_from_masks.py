import os
import cv2
import numpy as np
import argparse
import json
import h5py
import tqdm

def get_bbox_from_mask(mask):
    # If the mask is one-dimensional (1, height, width), squeeze it to (height, width)
    if len(mask.shape) == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)

    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return x, y, w, h

def pad_bbox(x, y, w, h, scale, image_shape):
    cx, cy = x + w / 2, y + h / 2
    new_w, new_h = w * scale, h * scale
    x1 = int(max(0, cx - new_w / 2))
    y1 = int(max(0, cy - new_h / 2))
    x2 = int(min(image_shape[1], cx + new_w / 2))
    y2 = int(min(image_shape[0], cy + new_h / 2))
    return x1, y1, x2, y2

def process_all_frames(frame_h5_path, mask_h5_path, output_dir, padding_scale, save_to_h5=False, save_to_png=True, file_suffix=None):
    os.makedirs(output_dir, exist_ok=True)
    offsets = {}

    # Determine suffix for output files
    suffix = f"_{file_suffix}" if file_suffix else ""

    # Open the HDF5 file for frames
    with h5py.File(frame_h5_path, 'r') as frame_h5:
        frames_dataset = frame_h5['frames']

        # Open the HDF5 file for masks
        with h5py.File(mask_h5_path, 'r') as mask_h5:
            masks_group = mask_h5["masks"]

            # If saving to HDF5, create an HDF5 file for cropped data
            if save_to_h5:
                h5_cropped_path = os.path.join(output_dir, f"cropped_data{suffix}.h5")
                h5_cropped_file = h5py.File(h5_cropped_path, "w")
                cropped_group = h5_cropped_file.create_group("cropped_frames")

            # Iterate through the frames
            for frame_idx, frame_name in tqdm.tqdm(enumerate(masks_group)):
                frame_group = masks_group[frame_name]
                frame = frames_dataset[frame_name]  # Get the corresponding frame
                # Combine all masks for the frame
                combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                for mask_name in frame_group:
                    mask = frame_group[mask_name][:]
                    combined_mask = np.maximum(combined_mask, mask)

                bbox = get_bbox_from_mask(combined_mask)
                if bbox is None:
                    print(f"❌ No person in mask: {frame_name}")
                    out_path = os.path.join(output_dir, f"{frame_name}_not_found.png")
                    # cv2.imwrite(out_path, cropped)
                    continue

                x, y, w, h = bbox
                x1, y1, x2, y2 = pad_bbox(x, y, w, h, padding_scale, frame.shape)
                cropped = frame[y1:y2, x1:x2]
                if save_to_png:
                    out_path = os.path.join(output_dir, f"{frame_name}.png")
                    cv2.imwrite(out_path, cropped)
                    print(f"✅ Saved PNG: {out_path}")

                if save_to_h5:
                    cropped_group.create_dataset(frame_name, data=cropped, dtype="uint8")

                offsets[frame_name] = {'x1': x1, 'y1': y1}

            # Close the HDF5 file if saving to HDF5
            if save_to_h5:
                h5_cropped_file.close()

    # Save crop offsets for later use
    offset_path = os.path.join(output_dir, f"crop_offsets{suffix}.json")
    with open(offset_path, "w") as f:
        json.dump(offsets, f)
    print(f"\n📄 Saved offset metadata to: {offset_path}")

# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop frames using masks and save padded bounding boxes.")
    parser.add_argument("--frame_h5", required=True, help="Path to HDF5 file containing frames.")
    parser.add_argument("--mask_h5", required=True, help="Path to HDF5 file containing masks.")
    parser.add_argument("--output_dir", required=True, help="Path to save cropped frames.")
    parser.add_argument("--padding_scale", type=float, default=1.2, help="Padding scale around bbox (default=1.2)")
    parser.add_argument("--save_to_h5", action="store_true", help="Save cropped data to an HDF5 file.")
    parser.add_argument("--save_to_png", action="store_true", help="Save cropped frames as PNG files.")
    parser.add_argument("--file_suffix", type=str, default=None, help="Suffix for output files (e.g., '01' for cropped_data_01.h5 and crop_offsets_01.json)")

    args = parser.parse_args()

    process_all_frames(
        frame_h5_path=args.frame_h5,
        mask_h5_path=args.mask_h5,
        output_dir=args.output_dir,
        padding_scale=args.padding_scale,
        save_to_h5=args.save_to_h5,
        save_to_png=args.save_to_png,
        file_suffix=args.file_suffix
    )
