import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
import h5py
import grounding_dino.groundingdino.datasets.transforms as T
import argparse
from PIL import Image
import tqdm
# Hyperparameters
TEXT_PROMPT = "walkway. "
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py" # GroundingDINO_SwinB_cfg.py
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth" # groundingdino_swinb_cogcoor.pth
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.
CONFIDENCE_FILTER = 0.3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Load models
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)

def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def preprocess_image(image_data):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    # convert image data to PIL Image for later processing
    if isinstance(image_data, np.ndarray):
        if image_data.dtype != np.uint8:
            image_data = image_data.astype(np.uint8)
        if image_data.ndim == 2:  # grayscale
            image_data = np.stack([image_data]*3, axis=-1)
        image_data = Image.fromarray(image_data)
    image_transformed, _ = transform(image_data, None)  # Directly assign the result
    return image_transformed

def process_frame(img, save_visualization=False, visualization_output_dir=None, frame_name=None):
    image_source = img
    image = preprocess_image(img)

    sam2_predictor.set_image(image_source)

    h, w, _ = image_source.shape

    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )

    # Apply confidence filter
    filtered = [(b, c, l) for b, c, l in zip(boxes, confidences, labels) if c >= CONFIDENCE_FILTER]
    if not filtered:
        print(f"No high-confidence objects found in frame {frame_name}")
        return {
            "annotations": [],
            "box_format": "xyxy",
            "img_width": w,
            "img_height": h,
        }

    boxes, confidences, labels = zip(*filtered)
    boxes = torch.stack(boxes)
    confidences = torch.tensor(confidences)

    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

    masks = masks.squeeze(1) if masks.ndim == 4 else masks

    rles = [single_mask_to_rle(mask) for mask in masks]
    result = {
        "annotations": [
            {
                "class_name": label,
                "bbox": box.tolist(),
                "segmentation": rle,
                "score": float(score),
            }
            for label, box, rle, score in zip(labels, input_boxes, rles, scores)
        ],
        "box_format": "xyxy",
        "img_width": w,
        "img_height": h,
    }

    # Optional visualization
    if save_visualization and visualization_output_dir and frame_name:
        detections = sv.Detections(
            xyxy=input_boxes,
            mask=masks.astype(bool),
            class_id=np.arange(len(labels))
        )

        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        mask_annotator = sv.MaskAnnotator()

        annotated = image_source.copy()
        annotated = box_annotator.annotate(annotated, detections)
        annotated = label_annotator.annotate(annotated, detections, labels=[f"{label} ({conf:.2f})" for label, conf in zip(labels, confidences)])
        annotated = mask_annotator.annotate(annotated, detections)

        out_path = os.path.join(visualization_output_dir, f"annotated_{frame_name}.jpg")
        cv2.imwrite(out_path, annotated)

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_h5", required=True, help="Path to output HDF5 file to save results.")
    parser.add_argument("--frame_h5", required=True, help="Path to input HDF5 file containing frame data.")
    args = parser.parse_args()

    output_h5_path = args.output_h5
    with h5py.File(output_h5_path, 'w') as h5_file:
        results_group = h5_file.create_group("results")

        with h5py.File(args.frame_h5, 'r') as h5_input:
            frames_dataset = h5_input['frames']
            # timestamps_dataset = h5_input['timestamps']

            for frame_idx, frame_name in tqdm.tqdm(enumerate(frames_dataset)):
                #print(f"\n📷 Processing frame {frame_idx + 1}/{len(frames_dataset)}")

                # Process the frame
                result = process_frame(frames_dataset[frame_name][:])

                # Save results to HDF5
                frame_group = results_group.create_group(f"{frame_name}")
                frame_group.create_dataset("annotations", data=json.dumps(result["annotations"]))
                frame_group.attrs["box_format"] = result["box_format"]
                frame_group.attrs["img_width"] = result["img_width"]
                frame_group.attrs["img_height"] = result["img_height"]

    print(f"\n✅ All results saved to {output_h5_path}")
