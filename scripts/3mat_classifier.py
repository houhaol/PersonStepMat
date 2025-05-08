import os
import argparse
import torch
import open_clip
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from glob import glob
import json
from collections import defaultdict
from sklearn.metrics import accuracy_score, classification_report
import random
from collections import deque, Counter


def load_and_split_support_set(support_dir, model, preprocess, device, val_ratio=0.2):
    """
    Load support images, compute features, and split into support/val.
    """
    all_embeddings = []
    all_labels = []
    label_names = sorted(os.listdir(support_dir))

    val_feats, val_labels = [], []
    support_feats, support_labels = [], []

    for label in label_names:
        image_paths = glob(os.path.join(support_dir, label, '*.jpg')) + \
                      glob(os.path.join(support_dir, label, '*.png'))
        random.shuffle(image_paths)
        split_idx = int(len(image_paths) * (1 - val_ratio))
        support_paths = image_paths[:split_idx]
        val_paths = image_paths[split_idx:]

        for img_path in support_paths:
            feature = extract_clip_feature(img_path, model, preprocess, device)
            support_feats.append(feature)
            support_labels.append(label)

        for img_path in val_paths:
            feature = extract_clip_feature(img_path, model, preprocess, device)
            val_feats.append(feature)
            val_labels.append(label)

    if val_ratio > 0.0:
        return (np.vstack(support_feats), support_labels,
        np.vstack(val_feats), val_labels)
    else:
        return (np.vstack(support_feats), support_labels)


def extract_clip_feature(img_path, model, preprocess, device):
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    image_tensor = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model.encode_image(image_tensor)
        feature /= feature.norm(dim=-1, keepdim=True)
    return feature.cpu().numpy()


def compute_prototypes(features, labels):
    label_to_features = defaultdict(list)
    for feat, label in zip(features, labels):
        label_to_features[label].append(feat)
    prototypes = {}
    for label, feats in label_to_features.items():
        feats = np.vstack(feats)
        proto = feats.mean(axis=0)
        proto /= np.linalg.norm(proto) + 1e-10
        prototypes[label] = proto
    return prototypes


def classify_with_prototypes(query_features, prototypes):
    preds = []
    for q_feat in query_features:
        best_score = -1
        best_label = None
        for label, proto in prototypes.items():
            score = np.dot(q_feat, proto)  # cosine similarity
            if score > best_score:
                best_score = score
                best_label = label
        preds.append(best_label)
    return preds


def evaluate_validation_set(val_feats, val_labels, prototypes):
    pred_labels = classify_with_prototypes(val_feats, prototypes)
    acc = accuracy_score(val_labels, pred_labels)
    print("\n🔍 Validation Accuracy: {:.2f}%".format(acc * 100))
    print("\n📊 Classification Report:")
    print(classification_report(val_labels, pred_labels, digits=3))


def predict_infer_dir_temporal(infer_dir, model, preprocess, prototypes, device,
                                window_size=5, ema_alpha=0.3, apply_ema=True,
                                output_json=None, save_json=False, save_csv=False):
    results = []
    vis_dir = output_json
    os.makedirs(vis_dir, exist_ok=True)

    # Sort image paths by filename (assuming filenames include timestamps)
    img_paths = sorted(glob(os.path.join(infer_dir, "*_rgb.png")))

    # Label list and mapping to/from indices
    label_list = sorted(prototypes.keys())
    label_to_idx = {label: i for i, label in enumerate(label_list)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}

    # Initialize EMA logits and sliding window for majority voting
    logits_ema = np.zeros(len(label_list))
    voting_window = deque(maxlen=window_size)

    for img_path in tqdm(img_paths):
        image = cv2.imread(img_path)
        patch_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(patch_rgb)
        image_tensor = preprocess(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            feature = model.encode_image(image_tensor)
            feature /= feature.norm(dim=-1, keepdim=True)

        query_feat = feature.cpu().numpy().squeeze()  # (D,)
        # Compute cosine similarities between query and all prototypes
        logits = np.array([np.dot(query_feat, prototypes[label]) for label in label_list])

        # Apply Exponential Moving Average (EMA) to logits
        if apply_ema:
            logits_ema = ema_alpha * logits + (1 - ema_alpha) * logits_ema
            smoothed_logits = logits_ema
        else:
            smoothed_logits = logits

        # Get the predicted class index from smoothed logits
        pred_idx = int(np.argmax(smoothed_logits))
        pred_label = idx_to_label[pred_idx]
        confidence = float(smoothed_logits[pred_idx])

        # Update the voting window and get the majority vote
        voting_window.append(pred_label)
        voted_label = Counter(voting_window).most_common(1)[0][0]

        # Store the results
        results.append({
            "image": os.path.basename(img_path),
            "label_raw": pred_label,
            "label_voted": voted_label,
            "confidence": confidence
        })

        # Visualization
        label_text = f"{voted_label} ({confidence:.2f})"
        annotated = image.copy()
        cv2.putText(annotated, label_text, (100, 100 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        vis_path = os.path.join(vis_dir, os.path.basename(img_path))
        cv2.imwrite(vis_path, annotated)

        print(f"🖼️ {os.path.basename(img_path)} → raw: {pred_label}, voted: {voted_label}")

    # Save all predictions to JSON file if needed
    if save_json and output_json:
        filename = os.path.join(output_json, "temporal_predictions.json")
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📄 Saved temporal predictions to: {filename}")

    # Save all predictions to csv file if needed
    if save_csv:
        filename = os.path.join(output_json, "temporal_predictions.csv")
        with open(filename, 'w') as f:
            f.write("image,label_raw,label_voted,confidence\n")
            for result in results:
                f.write(f"{result['image']},{result['label_raw']},{result['label_voted']},{result['confidence']}\n")
        print(f"\n📄 Saved temporal predictions to: {filename}")


        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP + Prototypical Networks with validation split")
    parser.add_argument("--support", required=True, help="Directory of support dataset organized by label")
    parser.add_argument("--infer", required=False, help="Directory of target inference frames")
    parser.add_argument("--model", default="ViT-B-32", help="OpenCLIP model name")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k", help="Pretrained model tag")
    parser.add_argument("--output", default="clip_material_predictions.json", help="Path to save predictions")
    parser.add_argument("--json", action="store_true", help="Save results to JSON")
    parser.add_argument("--csv", action="store_true", help="Save results to CSV")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Fraction of support samples to hold out for validation")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=args.model,
        pretrained=args.pretrained,
        device=device
    )
    model.eval()

    print("🔍 Loading support set and splitting into support/val...")
    if args.val_ratio > 0:
        support_feats, support_labels, val_feats, val_labels = load_and_split_support_set(
            args.support, model, preprocess, device, val_ratio=args.val_ratio
        )
    else:
        support_feats, support_labels = load_and_split_support_set(
            args.support, model, preprocess, device, val_ratio=0
        )

    print("📐 Computing prototypes from support set...")
    prototypes = compute_prototypes(support_feats, support_labels)

    if args.val_ratio > 0:
        print("🧪 Evaluating few-shot performance on held-out validation set...")
        evaluate_validation_set(val_feats, val_labels, prototypes)
    else:
        print("🔮 Running inference on target frames...")
        # predict_infer_dir(
        #     args.infer, model, preprocess,
        #     prototypes, device,
        #     output_json=args.output, save_json=args.json
        # )
        predict_infer_dir_temporal(
            args.infer, model, preprocess, prototypes, device,
            window_size=5,
            ema_alpha=0.4,
            apply_ema=True,
            output_json=args.output,
            save_json=args.json,
            save_csv=args.csv
        )

