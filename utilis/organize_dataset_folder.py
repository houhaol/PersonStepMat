import os
import shutil
import random

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../mat_support'))
dst_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../SGmaterials'))

splits = ['train', 'test']

# Get all class folders in src_dir
classes = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]

for cls in classes:
    class_dir = os.path.join(src_dir, cls)
    images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
    if not images:
        continue
    test_img = random.choice(images)
    for img in images:
        split = 'test' if img == test_img else 'train'
        dst_class_dir = os.path.join(dst_dir, split, cls)
        os.makedirs(dst_class_dir, exist_ok=True)
        shutil.copy2(os.path.join(class_dir, img), os.path.join(dst_class_dir, img))

print("Dataset organized successfully.")