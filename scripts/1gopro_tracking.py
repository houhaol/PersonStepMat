# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from https://github.com/facebookresearch/sam2/blob/main/notebooks/video_predictor_example.ipynb

# Organized imports
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
import h5py
from PIL import Image
from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor
import tqdm

def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    return device

def load_h5_data(h5_file):
    with h5py.File(h5_file, 'r') as f:
        frames_h5 = f['frames']
        image_data = np.array(frames_h5)

        # keys = list(frames_h5.keys())
        
        # # Get the shape of one frame to determine the dimensions
        # first_frame = frames_h5[keys[0]]
        # frame_shape = first_frame.shape  # (H, W, C)
        # num_frames = len(keys)
        
        # # Preallocate a numpy array for all frames
        # images_np = np.empty((num_frames, *frame_shape), dtype=first_frame.dtype)
        
        # # Iterate through each key and load the dataset into the preallocated array
        # for i, key in enumerate(keys):
        #     images_np[i] = frames_h5[key][...]  # Load the dataset directly into the array

    return image_data

def setup_model(device):
    checkpoint = "../checkpoints/efficienttam_s.pt"
    model_cfg = "configs/efficienttam/efficienttam_s.yaml"
    return build_efficienttam_video_predictor(model_cfg, checkpoint, device=device)

def propagate_video(predictor, inference_state):
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    return video_segments

def save_masks_to_h5(output_file, output_masks, image_data):
    with h5py.File(output_file, "w") as h5_file:
        masks_group = h5_file.create_group("masks")
        for out_frame_idx, frame_masks in enumerate(output_masks):

            frame_group = masks_group.create_group(str(image_data[out_frame_idx]))
            for obj_id, mask in enumerate(frame_masks):
                frame_group.create_dataset(f"mask_{obj_id}", data=mask, dtype="uint8")

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )

def render_segmentation_results(image_data, video_segments, output_dir=None, interval=100):
    if output_dir is None:
        return  # Skip rendering if no output directory is provided

    os.makedirs(output_dir, exist_ok=True)

    for out_frame_idx in range(0, image_data.shape[0], interval):
        plt.figure(figsize=(9, 6))
        plt.title(f"frame {out_frame_idx}")
        plt.imshow(str(image_data[out_frame_idx]))
        for out_obj_id, out_mask in video_segments.get(out_frame_idx, {}).items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        plt.axis("off")
        plt.savefig(os.path.join(output_dir, f"segmentation_frame_{out_frame_idx}.png"), bbox_inches="tight")
        plt.close()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process video frames and perform segmentation.")
    parser.add_argument("--h5_file", type=str, required=True, help="Path to the HDF5 file containing video frames.")
    parser.add_argument("--frame_idx", type=int, default=0, help="Index of the frame to visualize.")
    parser.add_argument("--output_file", type=str, default="mask_results.h5", help="Path to save the segmentation results.")
    parser.add_argument("--render_dir", type=str, default=None, help="Directory to save rendered segmentation results. If not provided, rendering is skipped.")
    return parser.parse_args()


def main():
    device = setup_device()
    args = parse_arguments()
    print("Loading the video data from HDF5 file...")
    image_data = load_h5_data(args.h5_file)
    print("Initializing video predictor...")
    predictor = setup_model(device)
    print("Initializing the state...")
    inference_state = predictor.init_state(video_path=args.h5_file, offload_video_to_cpu=True)

    # Let's now move on to the second object we want to track (giving it object id `3`)
    # with a positive click at (x, y) = (400, 150)
    points = np.array([[1000, 400], [1000,600], [1000,220]], dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1,1,1], np.int32)
    prompts = {3: (points, labels)}
    print("Adding new points or box at first frame")
    # `add_new_points_or_box` returns masks for all objects added so far on this interacted frame
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=3,
        points=points,
        labels=labels,
    )

    # show the results on the current (interacted) frame on all objects
    # plt.figure(figsize=(9, 6))
    # plt.title(f"frame {ann_frame_idx}")
    # plt.imshow(image_data[ann_frame_idx])
    # show_points(points, labels, plt.gca())
    # for i, out_obj_id in enumerate(out_obj_ids):
    #     show_points(*prompts[out_obj_id], plt.gca())
    #     show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
    # plt.savefig("segmentation_results.png", bbox_inches="tight")
    print("Starting video propagation...")
    video_segments = propagate_video(predictor, inference_state)

    # Define output_masks to store binary masks
    output_masks = []
    for out_frame_idx in tqdm.tqdm(range(len(image_data))):
        frame_masks = []
        for out_obj_id, out_mask in video_segments.get(out_frame_idx, {}).items():
            frame_masks.append(out_mask.astype(np.uint8))  # Convert mask to binary (0 or 1)
        output_masks.append(frame_masks)

    save_masks_to_h5(args.output_file, output_masks, image_data)
    render_segmentation_results(image_data, video_segments, output_dir=args.render_dir, interval=900)

if __name__ == "__main__":
    main()
