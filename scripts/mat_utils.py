import os
import random
import numpy as np
import torch
import cv2
from PIL import Image
from glob import glob
from collections import defaultdict

# MATERIAL_COLORS = {
#     "TerracottaTile": (80, 150, 220),
#     "Wood": (42, 42, 165),
#     "Dirt": (42, 42, 42),
#     "Metal": (0, 215, 255),
#     "Concrete": (80, 200, 100),
#     "VoidAcrylicPolymer": (128, 128, 0),
#     "Grass": (60, 179, 113),
#     "CeramicTiles": (255, 165, 0),
#     "RubberPlayground": (255, 128, 0),
#     "Asphalt": (128, 64, 128),
#     "RedCycling": (60, 20, 220),
#     "ConcretePavers": (100, 100, 200),
#     "GlazedBricks": (255, 255, 0),
#     "Unknown": (128, 128, 128),  # Default color for unknown materials
# }
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

def load_and_split_support_set_clip(support_dir, model, preprocess, device, val_ratio=0.2):
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

def extract_vit_feature(image, model, processor):
    img_np = np.array(image)
    # Create mask: non-black pixels (object)
    if img_np.ndim == 3:
        mask = np.any(img_np != 0, axis=2)
    else:
        mask = img_np != 0

    coords = np.argwhere(mask)
    if coords.size == 0:
        # fallback: use center crop
        crop = image.crop((0, 0, min(224, image.width), min(224, image.height)))
    else:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        crop = image.crop((x0, y0, x1, y1))
        # crop = crop.resize((224, 224))
    # coords = np.argwhere(mask)
    # if coords.size == 0:
    #     # fallback: use center crop
    #     crop = image.crop((0, 0, min(224, image.width), min(224, image.height)))
    # else:
    #     y0, x0 = coords.min(axis=0)
    #     y1, x1 = coords.max(axis=0) + 1
    #     region_h, region_w = y1 - y0, x1 - x0

    #     if region_h >= 224 and region_w >= 224:
    #         min_bg = None
    #         best_crop = None
    #         for i in range(y0, y1 - 224 + 1):
    #             for j in range(x0, x1 - 224 + 1):
    #                 window = mask[i:i+224, j:j+224]
    #                 bg = (window == 0).sum()
    #                 if (min_bg is None) or (bg < min_bg):
    #                     min_bg = bg
    #                     best_crop = (j, i, j+224, i+224)
    #         crop = image.crop(best_crop)
    #     else:
    #         crop = image.crop((x0, y0, x1, y1)).resize((224, 224))

    
    # inputs = processor(images=image, return_tensors="pt").to(model.device)
    inputs = processor(images=crop, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    feature = outputs.pooler_output
    feature = feature.cpu().detach().numpy()
    return feature

def load_and_split_support_set_hugging(support_dir, model, processor, device, val_ratio=0.2):
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
        # check file under support dir, if less than 10, omit this label
        if len(image_paths) < 10:
            continue
        random.shuffle(image_paths)
        split_idx = int(len(image_paths) * (1 - val_ratio))
        support_paths = image_paths[:split_idx]
        val_paths = image_paths[split_idx:]

        for img_path in support_paths:
            img = Image.open(img_path).convert("RGB")
            feature = extract_vit_feature(img, model, processor)
            support_feats.append(feature)
            support_labels.append(label)

        for img_path in val_paths:
            img = Image.open(img_path).convert("RGB")
            feature = extract_vit_feature(img, model, processor)
            # copy to cpu to avoid memory issues
            val_feats.append(feature)
            val_labels.append(label)

    if val_ratio > 0.0:
        return (np.vstack(support_feats), support_labels,
        np.vstack(val_feats), val_labels)
    else:
        return (np.vstack(support_feats), support_labels)


def extract_vit_feature_clip(image, model, processor):
    img_np = np.array(image)
    # Create mask: non-black pixels (object)
    if img_np.ndim == 3:
        mask = np.any(img_np != 0, axis=2)
    else:
        mask = img_np != 0

    coords = np.argwhere(mask)
    if coords.size == 0:
        # fallback: use center crop
        crop = image.crop((0, 0, min(224, image.width), min(224, image.height)))
    else:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        crop = image.crop((x0, y0, x1, y1))
        # crop = crop.resize((224, 224))

    outputs = model(processor(crop).unsqueeze(0))
    # import pdb; pdb.set_trace()
    # feature = outputs.pooler_output
    feature = outputs.cpu().detach().numpy()
    return feature

def extract_vit_feature_linear(image, model, processor):
    img_np = np.array(image)
    # Create mask: non-black pixels (object)
    if img_np.ndim == 3:
        mask = np.any(img_np != 0, axis=2)
    else:
        mask = img_np != 0

    coords = np.argwhere(mask)
    if coords.size == 0:
        # fallback: use center crop
        crop = image.crop((0, 0, min(224, image.width), min(224, image.height)))
    else:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        crop = image.crop((x0, y0, x1, y1))
        # crop = crop.resize((224, 224))

    img = Image.fromarray(np.array(crop))
    inp = processor(images=img, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inp, output_hidden_states=True)
        last_hidden = outputs.last_hidden_state.squeeze(0)  # (num_tokens, hidden_dim)
        class_token = last_hidden[0].cpu().numpy()  # (hidden_dim,)
        # patch_tokens = last_hidden[1:].cpu().numpy()  # (num_patches, hidden_dim)
        # avg_patch_token = patch_tokens.mean(axis=0)  # (hidden_dim,)
        num_reg = getattr(model.config, "num_register_tokens", 0)
        patch_tokens = last_hidden[1 + num_reg :].cpu().numpy()
        avg_patch_token = patch_tokens.mean(axis=0)  # (hidden_dim,)
        # normalize
        avg_patch_token = avg_patch_token / np.linalg.norm(avg_patch_token) if np.linalg.norm(avg_patch_token) > 0 else avg_patch_token
    return class_token, avg_patch_token

def extract_vit_feature_pipe(image, feature_extractor):
    img_np = np.array(image)
    # Create mask: non-black pixels (object)
    if img_np.ndim == 3:
        mask = np.any(img_np != 0, axis=2)
    else:
        mask = img_np != 0

    coords = np.argwhere(mask)
    if coords.size == 0:
        # fallback: use center crop
        crop = image.crop((0, 0, min(224, image.width), min(224, image.height)))
    else:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        crop = image.crop((x0, y0, x1, y1))
        # crop = crop.resize((224, 224))

    outputs = feature_extractor(crop)
    feature = np.array(outputs[0][0]).reshape(1, -1)
    return feature

def load_and_split_support_set_clip(support_dir, model, processor, device, val_ratio=0.2):
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
        # check file under support dir, if less than 10, omit this label
        if len(image_paths) < 10:
            continue
        random.shuffle(image_paths)
        split_idx = int(len(image_paths) * (1 - val_ratio))
        support_paths = image_paths[:split_idx]
        val_paths = image_paths[split_idx:]

        for img_path in support_paths:
            img = Image.open(img_path).convert("RGB")
            feature = extract_vit_feature_clip(img, model, processor)
            support_feats.append(feature)
            support_labels.append(label)

        for img_path in val_paths:
            img = Image.open(img_path).convert("RGB")
            feature = extract_vit_feature_clip(img, model, processor)
            # copy to cpu to avoid memory issues
            val_feats.append(feature)
            val_labels.append(label)

    if val_ratio > 0.0:
        return (np.vstack(support_feats), support_labels,
        np.vstack(val_feats), val_labels)
    else:
        return (np.vstack(support_feats), support_labels) 

def load_and_split_support_set_pipeline(support_dir, feature_extractor, device, val_ratio=0.2, threshold_10=True):
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
        # check file under support dir, if less than 10, omit this label
        if threshold_10 and len(image_paths) < 10:
            continue
        random.shuffle(image_paths)
        split_idx = int(len(image_paths) * (1 - val_ratio))
        support_paths = image_paths[:split_idx]
        val_paths = image_paths[split_idx:]

        for img_path in support_paths:
            img = Image.open(img_path).convert("RGB")
            feature = extract_vit_feature_pipe(img, feature_extractor)
            support_feats.append(feature)
            support_labels.append(label)

        for img_path in val_paths:
            img = Image.open(img_path).convert("RGB")
            feature = extract_vit_feature_pipe(img, feature_extractor)
            val_feats.append(feature)
            val_labels.append(label)

    if val_ratio > 0.0:
        return (np.vstack(support_feats), support_labels,
        np.vstack(val_feats), val_labels)
    else:
        return (np.vstack(support_feats), support_labels)

def load_and_split_support_set_linear(support_dir, model, processor,device, val_ratio=0.2, threshold_10=True):
    """
    Load support images, compute features, and split into support/val.
    """
    all_embeddings = []
    all_labels = []
    label_names = sorted(os.listdir(support_dir))

    val_class_tokens, val_patch_tokens = [], []
    val_labels = []
    support_class_tokens, support_patch_tokens = [], []
    support_labels = []
    for label in label_names:
        image_paths = glob(os.path.join(support_dir, label, '*.jpg')) + \
                      glob(os.path.join(support_dir, label, '*.png'))
        # check file under support dir, if less than 10, omit this label
        if threshold_10 and len(image_paths) < 10:
            continue
        random.shuffle(image_paths)
        split_idx = int(len(image_paths) * (1 - val_ratio))
        support_paths = image_paths[:split_idx]
        val_paths = image_paths[split_idx:]

        
        for img_path in support_paths:
            img = Image.open(img_path).convert("RGB")
            class_tok, patch_tok = extract_vit_feature_linear(img, model, processor)
            support_class_tokens.append(class_tok)
            support_patch_tokens.append(patch_tok)
            support_labels.append(label)

        
        for img_path in val_paths:
            img = Image.open(img_path).convert("RGB")
            class_tok, patch_tok = extract_vit_feature_linear(img, model, processor)
            val_class_tokens.append(class_tok)
            val_patch_tokens.append(patch_tok)
            val_labels.append(label)
    if val_ratio > 0.0:
        return (np.stack(support_class_tokens), np.stack(support_patch_tokens), support_labels,
        np.stack(val_class_tokens), np.stack(val_patch_tokens), val_labels)
    else:
        return (np.vstack(support_class_tokens), np.vstack(support_patch_tokens), support_labels)

def smooth_data(df, window_size=5):
    """
    Smooth material predictions using a mode filter for categorical data.

    Args:
        df (pd.DataFrame): DataFrame containing predictions with a 'pred_idx' column.
        window_size (int): Size of the sliding window.

    Returns:
        pd.DataFrame: DataFrame with an additional 'smoothed_pred_idx' column.
    """
    from collections import Counter

    def mode_filter(series):
        counts = Counter(series)
        return max(counts, key=counts.get) if counts else None

    # Apply mode filter over a sliding window
    df['smoothed_pred_idx'] = (
        df['pred_idx']
        .rolling(window=window_size, center=True)
        .apply(lambda x: mode_filter(x), raw=False)
    )

    # Fill NaN values at the start and end of the series with the nearest valid value
    # df['smoothed_pred_idx'] = df['smoothed_pred_idx'].fillna(method='bfill').fillna(method='ffill')
    df['smoothed_pred_idx'] = df['smoothed_pred_idx'].ffill().bfill()

    idx_to_label_raw = {row['pred_idx']: row['label_raw'] for _, row in df.drop_duplicates('pred_idx')[['pred_idx', 'label_raw']].iterrows()}
    df['label_smooth'] = df['smoothed_pred_idx'].map(idx_to_label_raw)
    return df

def convert_label_to_id(df):
    """
    Convert label_raw to numerical IDs based on MATERIAL_COLORS.
    """
    label_to_id = {label: idx for idx, label in enumerate(MATERIAL_COLORS.keys())}
    df['pred_idx'] = df['label_voted'].map(label_to_id)
    return df

def filter_data(df,confidence_threshold=0.8):
    """
    Filter DataFrame to only include rows with confidence above a threshold.
    """
    return df[df['confidence'] > confidence_threshold].reset_index(drop=True)