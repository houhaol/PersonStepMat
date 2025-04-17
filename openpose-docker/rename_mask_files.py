import os

mask_dir = "/home/houhao/workspace/Track-Anything/result/mask/upstairs_eye_tps/"  # replace with your mask folder

mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".npy")])

for i, old_name in enumerate(mask_files):
    new_name = f"mask_{i:04d}.npy"
    old_path = os.path.join(mask_dir, old_name)
    new_path = os.path.join(mask_dir, new_name)
    os.rename(old_path, new_path)
    print(f"✅ Renamed {old_name} → {new_name}")
