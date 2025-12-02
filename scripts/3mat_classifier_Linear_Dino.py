import os
import argparse
import torch
from transformers import pipeline
from transformers.image_utils import load_image
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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import h5py
from scipy.spatial.distance import cosine
from mat_utils import extract_vit_feature_linear,load_and_split_support_set_linear
from torch.nn.functional import cosine_similarity

from transformers import AutoImageProcessor, AutoModel


def evaluate_validation_set(pred_labels, val_labels):
    acc = accuracy_score(val_labels, pred_labels)
    print("\n🔍 Validation Accuracy: {:.2f}%".format(acc * 100))
    print("\n📊 Classification Report:")
    print(classification_report(val_labels, pred_labels, digits=3))

    # Plot confusion matrix as percentages
    labels = sorted(list(set(val_labels) | set(pred_labels)))
    cm = confusion_matrix(val_labels, pred_labels, labels=labels)
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    plt.figure(figsize=(10, 10))
    plt.imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=100)
    plt.title("Confusion Matrix (%)")
    plt.colorbar(label="Percentage")
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=30,ha='right')
    plt.yticks(tick_marks, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Add percentages inside the boxes
    thresh = cm_percent.max() / 2.
    for i in range(cm_percent.shape[0]):
        for j in range(cm_percent.shape[1]):
            plt.text(j, i, "{:.1f}%".format(cm_percent[i, j]),
                     ha="center", va="center",
                     color="white" if cm_percent[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig("val_confusion_matrix.png")

def get_accuracy(pred_labels, true_labels):
    acc = accuracy_score(true_labels, pred_labels)
    print("\n🔍 Validation Accuracy: {:.2f}%".format(acc * 100))
    return acc

def predict_infer_h5_temporal(frames, ground_mask, model, processor, clf, le, device,
                               window_size=5, output_json=None, save_json=False, save_csv=False, visualize=True):
    results = []
    vis_dir = output_json
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir, exist_ok=True)
    
    # Open the HDF5 file
    with h5py.File(frames, 'r') as h5_file, h5py.File(ground_mask, 'r') as ground_mask_file:
        frames = h5_file['frames']
        ground_masks = ground_mask_file['binary_masks']

        voting_window = deque(maxlen=window_size)

        for frame_id in tqdm(ground_masks):
            ground_mask = ground_masks[frame_id][:]
            frame = frames[frame_id][:]
            ground_mask = np.array(ground_mask)/ 255.0  # Normalize ground mask to [0, 1]
            frame = np.array(frame)
            # apply ground mask to frame
            if ground_mask.ndim == 2:  # If ground mask is 2D
                ground_mask = np.expand_dims(ground_mask, axis=-1)  # Make it 3D
            if ground_mask.shape[-1] == 1:  # If single channel, convert to 3 channels
                ground_mask = np.repeat(ground_mask, 3, axis=-1)
            # Apply ground mask to frame
            patch_rgb = (frame * ground_mask).astype(np.uint8)
            patch_rgb = np.clip(patch_rgb, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(patch_rgb)

            query_class_token, query_patch_token = extract_vit_feature_linear(pil_img, model, processor)
            # Ensure both tokens are 2D arrays for concatenation
            query_feature = np.concatenate([query_class_token, query_patch_token])
            query_feature = query_feature.reshape(1, -1)  # Reshape to 2D array for prediction
            y_pred = clf.predict_proba(query_feature)
            pred_label = le.inverse_transform(y_pred.argmax(axis=1))[0]

            confidence = y_pred.max()
            # Update the voting window and get the majority vote
            voting_window.append(pred_label)
            voted_label = Counter(voting_window).most_common(1)[0][0]

            # Store the results
            results.append({
                "frame": frame_id,
                "label_raw": pred_label,
                "label_voted": voted_label,
                "confidence": confidence,
            })

            # Visualization
            if visualize:
                label_text = f"{voted_label} ({confidence:.2f})"
                annotated = frame.copy()
                cv2.putText(annotated, label_text, (100, 100 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                vis_path = os.path.join(vis_dir, f"{frame_id}.png")
                cv2.imwrite(vis_path, annotated)

                print(f"🖼️ {frame_id} → raw: {pred_label}, voted: {voted_label}")

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
            f.write("frame,label_raw,label_voted,confidence\n")
            for result in results:
                f.write(f"{result['frame']},{result['label_raw']},{result['label_voted']},{result['confidence']}\n")
        print(f"\n📄 Saved temporal predictions to: {filename}")


def train_linear_classifier(support_class_tokens, support_patch_tokens, support_labels, val_class_tokens=None, val_patch_tokens=None, val_labels=None):
    """
    Trains a linear classifier on concatenated ViT tokens. Optionally evaluates on validation set.
    Returns: trained classifier, label encoder
    """
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression
    X_train = np.concatenate([support_class_tokens, support_patch_tokens], axis=1)
    y_train = np.array(support_labels)
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, y_train_enc)

    if val_class_tokens is not None and val_patch_tokens is not None and val_labels is not None:
        X_val = np.concatenate([val_class_tokens, val_patch_tokens], axis=1)
        y_val = np.array(val_labels)
        y_pred = clf.predict_proba(X_val)
        pred_labels = le.inverse_transform(y_pred.argmax(axis=1))
        return clf, le, pred_labels
        # evaluate_validation_set(pred_labels, y_val)
    return clf, le


def curve_plot():
    val_ratios = [i for i in np.arange(0.1, 1.0, 0.1)]
    # val_ratios = [0.2]
    label_names = sorted(os.listdir(args.support))
    total_files = 0
    for label in label_names:
        image_paths = glob(os.path.join(args.support, label, '*.jpg')) + \
                        glob(os.path.join(args.support, label, '*.png'))
        total_files += len(image_paths)

    accuracy_record = []
    for val_ratio in val_ratios:
        tmp_accuracy = []
        for i in range(10):
            print("🔍 Loading support set and splitting into support/val by {}...".format(val_ratio))
            support_class_tokens, support_patch_tokens, support_labels, val_class_tokens, val_patch_tokens, val_labels = load_and_split_support_set_linear(
                args.support, model, processor, device, val_ratio=val_ratio, threshold_10=True
            )
            print("\n🔄 Training linear classifier and evaluating on validation set...")
            clf, le, pred_labels = train_linear_classifier(
                support_class_tokens, support_patch_tokens, support_labels,
                val_class_tokens, val_patch_tokens, val_labels
            )
            tmp_accuracy.append(get_accuracy(pred_labels, np.array(val_labels)))
        accuracy_record.append(np.mean(tmp_accuracy))
        # evaluate_validation_set(pred_labels, np.array(val_labels))

    val_files_used = [int(i*total_files) for i in val_ratios]
    # line plot the accuracy_record with respect to val_ratios, x label should be ratio + val_files_used
    plt.plot(val_files_used, accuracy_record, marker='o')
    plt.xlabel("Validation Files Used")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Validation Files Used")
    plt.grid()
    plt.savefig("accuracy_vs_validation_files.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViT + Linear Classifier with validation split (Hugging Face pipeline version)")
    parser.add_argument("--support", required=True, help="Directory of support dataset organized by label")
    parser.add_argument('--ground_mask', help="Directory containing ground mask images")
    parser.add_argument('--frame_name', help="Directory containing frame images")
    parser.add_argument("--model", default="google/vit-base-patch16-224", help="Hugging Face ViT model name")
    parser.add_argument("--output", default="clip_material_predictions.json", help="Path to save predictions")
    parser.add_argument("--json", action="store_true", help="Save results to JSON")
    parser.add_argument("--csv", action="store_true", help="Save results to CSV")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Fraction of support samples to hold out for validation")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization of predictions")
    parser.add_argument("--curve", action="store_true", help="Enable curve visualization by adjusting val_ratio")
    args = parser.parse_args()

    device = 0 if torch.cuda.is_available() else -1
    name = "facebook/dinov3-vits16-pretrain-lvd1689m"
    # name = "google/vit-base-patch16-224"
    processor = AutoImageProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name).eval().to("cuda")

    if args.curve:
        curve_plot()
    else:
        print("Inference Started")
        # if feature set exist load else create
        if os.path.exists("../materials/SGWalkwayMaterials_feature_set.npz"):
            data = np.load("../materials/SGWalkwayMaterials_feature_set.npz")
            support_class_tokens = data['class_tokens']
            support_patch_tokens = data['patch_tokens']
            support_labels = data['labels']
            print("Loaded the precomputed materials features")
        else:
            print("Preparing the features")
            support_class_tokens, support_patch_tokens, support_labels = load_and_split_support_set_linear(
                    args.support, model, processor, device, val_ratio=0
                )
            # Save the support tokens and labels for later use
            np.savez("../materials/SGWalkwayMaterials_feature_set.npz", class_tokens=support_class_tokens, patch_tokens=support_patch_tokens, labels=support_labels)

        print("\n🔄 Training linear classifier...")
        clf, le = train_linear_classifier(support_class_tokens, support_patch_tokens, support_labels)

        print("🔮 Running inference on target frames...")
        predict_infer_h5_temporal(
            args.frame_name, args.ground_mask, model, processor, clf, le, device,
            window_size=5,
            output_json=args.output,
            save_json=args.json,
            save_csv=args.csv,
            visualize=args.visualize
        )


