from email.mime import image
import h5py
import numpy as np
import json
from pycocotools import mask as maskUtils
import cv2
import os
import tqdm
import pandas as pd
import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# OpenPose BODY_25 connections
BODY_25_PAIRS_RENDER = [
    (1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
    (0, 1), (0, 15), (15, 17), (0, 16), (16, 18),
    (14, 21), (11, 22), (14, 19), (19, 20), (11, 24), (22, 23)
]

POSE_COLORS = [
    (255, 0, 85), (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170),
    (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
    (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85),
    (128, 128, 255), (128, 255, 128), (255, 128, 128), (255, 255, 128)
]

MATERIAL_COLORS = {
    "AcrylicPolymer": (31, 119, 180),     # Blue (tab20 color 0)
    "Asphalt": (44, 160, 44),             # Green (tab20 color 4)
    "BicycleCoat": (214, 39, 40),         # Red (tab20 color 6)
    "Bricks": (255, 127, 14),             # Orange (tab20 color 2)
    "Brushed_Concrete_New": (148, 103, 189), # Purple (tab20 color 8)
    "Brushed_Concrete_Old": (140, 86, 75),   # Brown (tab20 color 10)
    "Ceramic_Porcelain_Tiles": (227, 119, 194), # Pink (tab20 color 12)
    "ConcretePavers": (127, 127, 127),    # Gray (tab20 color 14)
    "Dirt": (188, 189, 34),               # Olive (tab20 color 16)
    "Exposed_Aggregate_Concrete": (23, 190, 207), # Cyan (tab20 color 18)
    "Granite": (174, 199, 232),           # Light Blue (tab20 color 1)
    "Grass": (152, 223, 138),             # Light Green (tab20 color 5)
    "Metal": (255, 187, 120),             # Light Orange (tab20 color 3)
    "Rubber": (197, 176, 213),            # Light Purple (tab20 color 9)
    "Unknown": (196, 156, 148),           # Light Brown (tab20 color 11)
}

def draw_keypoints(image, keypoints, point_radius=5, line_thickness=2):
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.05:
            cv2.circle(image, (int(x), int(y)), point_radius, (0, 255, 255), -1)

    # Skip head connections (neck to head connections)
    head_connections = [(0, 1), (0, 15), (15, 17), (0, 16), (16, 18), (14,19)]
    
    for idx, (a, b) in enumerate(BODY_25_PAIRS_RENDER):
        # Skip head connections
        if (a, b) in head_connections or (b, a) in head_connections:
            continue
            
        ptA = (int(keypoints[a][0]), int(keypoints[a][1]))
        ptB = (int(keypoints[b][0]), int(keypoints[b][1]))
        color = POSE_COLORS[idx % len(POSE_COLORS)]
        cv2.line(image, ptA, ptB, color, thickness=line_thickness)

    # draw foot
    foot_prompts = []
    shift_y = 10  # Adjust this value as needed
    # Foot prompts - left
    left_indices = [21, 19, 20]
    left_points = [
        {"foot": "left", "point": [int(keypoints[idx][0]), int(keypoints[idx][1] + shift_y)]}
        for idx in left_indices if keypoints[idx][2] > 0.1
    ]
    if left_points:
        # Store only the one with lowest y (i.e., largest y value)
        lowest_left = max(left_points, key=lambda p: p["point"][1])
        foot_prompts.append(lowest_left)

    # Foot prompts - right
    right_indices = [24, 22, 23]
    right_points = [
        {"foot": "right", "point": [int(keypoints[idx][0]), int(keypoints[idx][1] + shift_y)]}
        for idx in right_indices if keypoints[idx][2] > 0.1
    ]
    if right_points:
        lowest_right = max(right_points, key=lambda p: p["point"][1])
        foot_prompts.append(lowest_right)

    for prompt in foot_prompts:
        px, py = prompt["point"]
        cv2.circle(image, (px, py+20), 6, (0, 0, 255), -1)
        cv2.putText(image, prompt["foot"], (px + 5, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return image

def load_material_labels_csv(csv_file):
    df = pd.read_csv(csv_file)
    by_frame = df.set_index("frame", verify_integrity=True)
    # Assumes columns: frame,label, confidence
    return by_frame

def inspect_raw_frames(h5_file, debug_dir):
    with h5py.File(h5_file, 'r') as h5_file:
        frames_dataset = h5_file['frames']
        # unravel the dataset to get the array of frames, like (N, H, W, C)
        frames_array = np.array(frames_dataset)
        # now i get the keys of the array, which are the frame names
        frame_names = list(frames_dataset.keys())

        # sample 10 images from the frames and save for debug purpose
        n = len(frame_names)
        module = n // 50
        for i in tqdm.tqdm(range(0, n, module)):
            frame_name = frame_names[i]
            frame_data = frames_dataset[frame_name][:]
            # Save the frame as an image file
            cv2.imwrite(f"{debug_dir}/debug_frame_{i}.png", cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR))
    print("Finished saving debug frames.")
    return frames_array, frame_names

def inspect_cropped_data(cropped_h5_path, debug_dir):
    with h5py.File(cropped_h5_path, 'r') as cropped_h5:
        cropped_data = cropped_h5['cropped_frames']
        frame_names = list(cropped_data.keys())
        n = len(frame_names)
        module = n // 10
        for i in tqdm.tqdm(range(0, n, module)):
            frame_name = frame_names[i]
            crop_patch = cropped_data[frame_name][()]
            # save the cropped patch as image for debug purpose
            cv2.imwrite(f"{debug_dir}/debug_cropped_{i}.png", cv2.cvtColor(crop_patch, cv2.COLOR_RGB2BGR))
    print("Finished saving cropped data info.")
    return cropped_data

def inspect_openpose_tracking(frames_h5_path, openpose_h5_path, cropped_data, offset_json, debug_dir):
    with open(offset_json, "r") as f:
        crop_offsets = json.load(f)
    with h5py.File(frames_h5_path, 'r') as frames_h5, h5py.File(openpose_h5_path, 'r') as openpose_h5:
        frames = frames_h5['frames']
        frames_names = list(frames.keys())
        n = len(frames_names)
        module = n // 10
        for i in tqdm.tqdm(range(0, n, module)):
            frame_name = frames_names[i]
            frame_rgb = frames[frame_name][:]
            try:
                keypoints = openpose_h5['pose_keypoints'][frame_name][:]
            except:
                continue
            # Load offsets
            offset = crop_offsets.get(frame_name, {"x1": 0, "y1": 0})
            dx, dy = offset["x1"], offset["y1"]
            keypoints[:, 0] += dx
            keypoints[:, 1] += dy

            image = draw_keypoints(frame_rgb, keypoints)

            out_path = os.path.join(debug_dir, f"{frame_name}.png")
            cv2.imwrite(out_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def inspect_pp_tracking(frames_h5_path, masks_h5_path, debug_dir):
    with h5py.File(frames_h5_path, 'r') as frames_h5, h5py.File(masks_h5_path, 'r') as masks_h5:
        frames = frames_h5['frames']
        masks = masks_h5['masks']
        frames_names = list(frames.keys())
        n = len(frames_names)
        module = n // 10
        for i in tqdm.tqdm(range(0, n, module)):
            frame_name = frames_names[i]
            frame_rgb = frames[frame_name][:]
            mask = masks[frame_name]['mask_0'][:]

            # Ensure mask is binary and 3-channel for overlay
            mask = (mask > 0).astype(np.uint8) * 255
            mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            red_mask = np.zeros_like(frame_rgb)
            red_mask[:, :, 2] = mask  # Red channel

            overlay = cv2.addWeighted(frame_rgb, 0.7, red_mask, 0.3, 0)
            out_path = os.path.join(debug_dir, f"{frame_name}.png")
            cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            # print(f"Saved overlay: {out_path}")

def inspect_ground_masks(ground_h5_path, frames_h5_path, debug_dir):
     # Open the HDF5 files
    with h5py.File(ground_h5_path, 'r') as ground_h5, h5py.File(frames_h5_path, 'r') as frame_h5:
        ground_h5 = ground_h5['results']
        frames_dataset = frame_h5['frames']
        frame_names = list(frames_dataset.keys())
        n = len(frame_names)
        module = n // 10
        for i in tqdm.tqdm(range(0, n, module)):
            frame_name = frame_names[i]
            frame_data = frames_dataset[frame_name][:]

            grounds = ground_h5[frame_name]
            annotations = json.loads(grounds['annotations'][()])
            overlay = frame_data.copy()

            # Generate distinct colors for each segmentation
            num_anns = len(annotations)
            colors = [
                tuple(np.random.randint(0, 256, 3).tolist())
                for _ in range(num_anns)
            ]

            for idx, ann in enumerate(annotations):
                seg = ann['segmentation']
                rle = {
                    'counts': seg['counts'],
                    'size': seg['size']
                }
                mask = maskUtils.decode(rle)
                mask = (mask > 0).astype(np.uint8) * 255
                color = colors[idx]
                color_mask = np.zeros_like(frame_data, dtype=np.uint8)
                for c in range(3):
                    color_mask[:, :, c] = (mask // 255) * color[c]

                # Overlay the mask on the image
                overlay = cv2.addWeighted(overlay, 1.0, color_mask, 0.5, 0)

            output_path = os.path.join(debug_dir, f"{frame_name}.png")
            # Convert BGR to RGB for saving
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            cv2.imwrite(output_path, overlay)
            print(f"✅ Saved visualization for {frame_name} to {output_path}")

def inspect_ground_masks_filtered(ground_h5_path, frames_h5_path, debug_dir):
    # Open the HDF5 files
    with h5py.File(ground_h5_path, 'r') as output_h5, h5py.File(frames_h5_path, 'r') as frame_h5:
        binary_masks = output_h5['binary_masks']
        frames_dataset = frame_h5['frames']

        frame_names = list(frames_dataset.keys())
        n = len(frame_names)
        module = n // 10
        for i in tqdm.tqdm(range(0, n, module)):
            try:
                frame_name = frame_names[i]
                frame_data = frames_dataset[frame_name][:]

                # Load the binary mask and the corresponding frame
                mask = binary_masks[frame_name][:]

                # Ensure the mask is binary and uint8
                mask = (mask > 0).astype(np.uint8) * 255

                # Convert mask to 3 channels for overlay
                mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

                # Overlay the mask on the frame (e.g., mask in red)
                overlay = frame_data.copy()
                red_mask = np.zeros_like(frame_data)
                red_mask[:, :, 2] = mask  # Red channel
                # Blend the original frame and the red mask
                overlay = cv2.addWeighted(overlay, 1.0, red_mask, 0.3, 0)

                # Save the visualized frame
                output_path = os.path.join(debug_dir, f"{frame_name}.png")
                # convert bgr to rgb
                overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                cv2.imwrite(output_path, overlay)
                print(f"✅ Saved visualization for {frame_name} to {output_path}")
            except:
                continue

def inspect_materials_classification(frames_h5_path, ground_h5_path, mat_file, label, debug_dir):
    # check if mat file is csv. 
    if mat_file.endswith(".csv"):
        # Load material labels from CSV. 
        material_labels = load_material_labels_csv(mat_file)
    else:
        print("No csv file found")
        return

    # Open the HDF5 files
    with h5py.File(ground_h5_path, 'r') as output_h5, h5py.File(frames_h5_path, 'r') as frame_h5:
        binary_masks = output_h5['binary_masks']
        frames_dataset = frame_h5['frames']

        frame_names = list(frames_dataset.keys())
        n = len(frame_names)
        module = n // 10
        for i in tqdm.tqdm(range(0, n, module)):
            frame_name = frame_names[i]
            frame_data = frames_dataset[frame_name][:]

            # Load the binary mask and the corresponding frame
            mask = binary_masks[frame_name][:]

            # Ensure the mask is binary and uint8
            mask = (mask > 0).astype(np.uint8) * 255

            # Convert mask to 3 channels for overlay
            mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # Overlay the mask on the frame (e.g., mask in red)
            overlay = frame_data.copy()
            red_mask = np.zeros_like(frame_data)
            red_mask[:, :, 2] = mask  # Red channel
            # Blend the original frame and the red mask
            overlay = cv2.addWeighted(overlay, 1.0, red_mask, 0.3, 0)

            # Get Material label
            material_label = material_labels.loc[frame_name, label]
            material_confidence = material_labels.loc[frame_name, 'confidence']
            
            # Get color for the material
            material_color = MATERIAL_COLORS.get(material_label, (128, 128, 128))  # Default to gray if not found
            
            # Create colored mask overlay
            colored_mask = np.zeros_like(frame_data)
            colored_mask[:, :, 0] = (mask // 255) * material_color[0]
            colored_mask[:, :, 1] = (mask // 255) * material_color[1]
            colored_mask[:, :, 2] = (mask // 255) * material_color[2]
            
            # Blend the original frame with the colored mask
            overlay = cv2.addWeighted(frame_data, 0.7, colored_mask, 0.3, 0)
            
            # Add text label
            font_scale = 2
            thickness = 4
            text = material_label + f" ({material_confidence:.2f})"
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            img_h, img_w = overlay.shape[:2]
            x = (img_w - text_width) // 2
            y = (img_h - 200)
            cv2.putText(overlay, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)

            # Save the visualized frame with overlay
            output_path = os.path.join(debug_dir, f"{frame_name}.png")
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            cv2.imwrite(output_path, overlay)
            
            # Save the masked segmented image with material color
            masked_image = np.zeros_like(frame_data)
            masked_image[:, :, 0] = (mask // 255) * material_color[0]
            masked_image[:, :, 1] = (mask // 255) * material_color[1]
            masked_image[:, :, 2] = (mask // 255) * material_color[2]
            
            masked_output_path = os.path.join(debug_dir, f"{frame_name}_masked.png")
            masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(masked_output_path, masked_image)
            
            print(f"✅ Saved visualization for {frame_name} to {output_path} and {masked_output_path}")

def load_width_labels_csv(csv_file):
    df = pd.read_csv(csv_file)
    by_frame = df.set_index("frame_name", verify_integrity=True)
    # Assumes columns: frame,width_m
    return by_frame

def inspect_width_measurements(frames_h5_path, ground_h5_path, width_file, label, debug_dir):
    # check if width file is csv. 
    if width_file.endswith(".csv"):
        # Load material labels from CSV. 
        width_labels = load_width_labels_csv(width_file)
    else:
        print("No csv file found")
        return

    # Open the HDF5 files
    with h5py.File(ground_h5_path, 'r') as output_h5, h5py.File(frames_h5_path, 'r') as frame_h5:
        binary_masks = output_h5['binary_masks']
        frames_dataset = frame_h5['frames']

        frame_names = list(frames_dataset.keys())
        n = len(frame_names)
        module = n // 10
        for i in tqdm.tqdm(range(0, n, module)):
            frame_name = frame_names[i]
            frame_data = frames_dataset[frame_name][:]

            # Load the binary mask and the corresponding frame
            try:
                mask = binary_masks[frame_name][:]
            except KeyError:
                print(f"Warning: Mask for frame {frame_name} not found.")
                continue

            # Ensure the mask is binary and uint8
            mask = (mask > 0).astype(np.uint8) * 255

            # Convert mask to 3 channels for overlay
            mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # Overlay the mask on the frame (e.g., mask in red)
            overlay = frame_data.copy()
            red_mask = np.zeros_like(frame_data)
            red_mask[:, :, 2] = mask  # Red channel
            # Blend the original frame and the red mask
            overlay = cv2.addWeighted(overlay, 1.0, red_mask, 0.3, 0)

            # Get Material label
            width = width_labels.loc[frame_name, label]
            font_scale = 2
            thickness = 4
            text = f"Width: {width:.2f} m"
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            img_h, img_w = overlay.shape[:2]
            x = (img_w - text_width) // 2
            y = (img_h - 200)
            cv2.putText(overlay, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)

            # Save the visualized frame
            output_path = os.path.join(debug_dir, f"{frame_name}.png")
            # convert bgr to rgb
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            cv2.imwrite(output_path, overlay)
            print(f"✅ Saved visualization for {frame_name} to {output_path}")

def run_mask2former(images, processor, model, target_sizes=None):
    # images: list of PIL Images or a single PIL Image
    if not isinstance(images, list):
        images = [images]
    inputs = processor(images=images, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    if target_sizes is None:
        target_sizes = [img.size[::-1] for img in images]
    segs = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
    # Convert each segmentation map to numpy array
    segs = [seg.cpu().numpy() for seg in segs]
    return segs if len(segs) > 1 else segs[0]

def get_label_id(label_name, id2label):
    for id, name in id2label.items():
        if name.lower() == label_name.lower():
            return id
    return None

def get_greenery_label_ids(id2label):
    greenery_keywords = ["tree", "grass", "vegetation", "bush", "plant", "shrub", "foliage"]
    greenery_ids = []
    for id, name in id2label.items():
        for keyword in greenery_keywords:
            if keyword in name.lower():
                greenery_ids.append(id)
                break
    return greenery_ids

def sky_greenery_ratio(segmentation_map, id2label):
    # compute sky and greenery ratio given segmentation map and id2label mapping
    total_pixels = segmentation_map.size
    sky_pixels = np.sum(segmentation_map == get_label_id("sky", id2label))
    greenery_pixels = np.sum(np.isin(segmentation_map, get_greenery_label_ids(id2label)))
    sky_ratio = sky_pixels / total_pixels
    greenery_ratio = greenery_pixels / total_pixels
    return sky_ratio, greenery_ratio

def sky_greenery_ratio_analysis(frames_h5_path, debug_dir):
    from PIL import Image
    model_name = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
    # id2label_path = "./mask2former_id2label.txt"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
    model.to(DEVICE)
    id2label = model.config.id2label

    with h5py.File(frames_h5_path, 'r') as frames_h5:
        frames = frames_h5['frames']
        frames_names = list(frames.keys())
        n = len(frames_names)
        module = n // 20
        for i in tqdm.tqdm(range(0, n, module)):
            frame_name = frames_names[i]
            frame_rgb = frames[frame_name][:]
            image = Image.fromarray(frame_rgb.astype(np.uint8)).convert("RGB")
            seg = run_mask2former(image, processor, model)

            # Compute sky and greenery ratio
            sky_ratio, greenery_ratio = sky_greenery_ratio(seg, id2label)
            # only visualize sky and greenery
            sky_label_id = get_label_id("sky", id2label)
            greenery_label_ids = get_greenery_label_ids(id2label)
            color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
            if sky_label_id is not None:
                color_seg[seg == sky_label_id, :] = [0,0,255]  # Red for sky
            if greenery_label_ids:
                color_seg[np.isin(seg, greenery_label_ids), :] = [0, 255, 0]  # Green for greenery

            image = cv2.addWeighted(frame_rgb, 0.7, color_seg, 0.3, 0)
            out_path = os.path.join(debug_dir, f"{frame_name}.png")
            cv2.putText(image, f"Sky: {sky_ratio:.4f}, Greenery: {greenery_ratio:.4f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            cv2.imwrite(out_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    # Path to the HDF5 file
    # suffix = '_01'
    # frames_h5_path = f"/media/houhao/BEFIT/PersonStepMat/BF007/frames{suffix}.h5"
    # masks_h5_path = f"/media/houhao/BEFIT/PersonStepMat/BF007/pp_masks{suffix}.h5"

    # openpose_h5_path = f"/media/houhao/BEFIT/PersonStepMat/BF007/cropped/openpose{suffix}.h5"
    # cropped_h5_path = f"/media/houhao/BEFIT/PersonStepMat/BF007/cropped/cropped_data{suffix}.h5"
    # offset_json = f"/media/houhao/BEFIT/PersonStepMat/BF007/cropped/crop_offsets{suffix}.json"

    # ground_h5_path = f"/media/houhao/BEFIT/PersonStepMat/BF007/ground{suffix}.h5"

    # filter_ground_h5_path = f"/media/houhao/BEFIT/PersonStepMat/BF007/filter_ground{suffix}.h5"

    # mat_file = f"/media/houhao/BEFIT/PersonStepMat/BF007/mat_pred{suffix}/mat_predictions{suffix}.csv"
    # width_file = f"/media/houhao/BEFIT/PersonStepMat/BF007/walkway_width_smoothed{suffix}.csv"

    # debug_dir = "/home/houhao/workspace/PersonStepMat/dataset/BF007/frames_debug"

    suffix = '_03'
    BF_ID = 'BF008'
    frames_h5_path = f"/home/houhao/workspace/PersonStepMat/dataset/{BF_ID}/frames{suffix}.h5"
    masks_h5_path = f"/home/houhao/workspace/PersonStepMat/dataset/{BF_ID}/pp_masks{suffix}.h5"

    openpose_h5_path = f"/home/houhao/workspace/PersonStepMat/dataset/{BF_ID}/cropped/openpose{suffix}.h5"
    cropped_h5_path = f"/home/houhao/workspace/PersonStepMat/dataset/{BF_ID}/cropped/cropped_data{suffix}.h5"
    offset_json = f"/home/houhao/workspace/PersonStepMat/dataset/{BF_ID}/cropped/crop_offsets{suffix}.json"

    ground_h5_path = f"/home/houhao/workspace/PersonStepMat/dataset/{BF_ID}/ground{suffix}.h5"

    filter_ground_h5_path = f"/home/houhao/workspace/PersonStepMat/dataset/{BF_ID}/filter_ground{suffix}.h5"

    mat_file = f"/home/houhao/workspace/PersonStepMat/dataset/{BF_ID}/mat_pred{suffix}/mat_predictions{suffix}.csv"
    width_file = f"/home/houhao/workspace/PersonStepMat/dataset/{BF_ID}/walkway_width_smoothed{suffix}.csv"

    debug_dir = f"/home/houhao/workspace/PersonStepMat/dataset/{BF_ID}/frames_debug"

    # if debug_dir not exist
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    # inspect gopro convert to rgb video frames
    frames_array, frame_names = inspect_raw_frames(frames_h5_path, debug_dir)

    # inspect tracking on participants
    # inspect_pp_tracking(frames_h5_path, masks_h5_path, debug_dir)

    # inspect cropped data
    # cropped_data = inspect_cropped_data(cropped_h5_path, debug_dir)

    # inspect openpose results on participants
    # inspect_openpose_tracking(frames_h5_path, openpose_h5_path, cropped_h5_path, offset_json, debug_dir)

    # inspect dino+sam predicted grounds
    # inspect_ground_masks(ground_h5_path, frames_h5_path, debug_dir)

    # inspect foot filtered predicted grounds
    # inspect_ground_masks_filtered(filter_ground_h5_path, frames_h5_path, debug_dir)

    # inspect materials classification. label option is label_raw, label_voted
    # inspect_materials_classification(frames_h5_path, filter_ground_h5_path, mat_file, label='label_voted', debug_dir=debug_dir)

    # inspect width
    # inspect_width_measurements(frames_h5_path, filter_ground_h5_path, width_file, label='width_m', debug_dir=debug_dir)

    # sky and greenery ratio analysis
    # sky_greenery_ratio_analysis(frames_h5_path, debug_dir)


