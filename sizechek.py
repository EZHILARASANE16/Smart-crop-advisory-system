from PIL import Image
import os

root_dir = r"C:\Users\Godwin Arulraj\Desktop\sih2025\data\PlantVillage"

sizes = {}

# loop over all folders and files
for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(folder_path, filename)
                img = Image.open(img_path)
                sizes[img.size] = sizes.get(img.size, 0) + 1
                break  # just check 1 image per folder for quick results
        print(f"Checked: {folder}")

print("\nUnique image sizes found:")
for size, count in sizes.items():
    print(f"{size} → {count} samples (at least)")
