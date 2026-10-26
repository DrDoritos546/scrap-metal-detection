import os
import random
import shutil

# Target the exact nested directories where your data lives
src_dir = "/workspace/copperdetection"
dest_dir = "/workspace/scrap_dataset"

image_folder = os.path.join(src_dir, "images", "train")
label_folder = os.path.join(src_dir, "labels", "train")

# Rebuild clean target directories
for split in ["train", "val"]:
    os.makedirs(os.path.join(dest_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, split, "labels"), exist_ok=True)

# Gather all files from the nested folder
all_files = os.listdir(image_folder)
images = [f for f in all_files if os.path.isfile(os.path.join(image_folder, f))]

if len(images) == 0:
    print(f"CRITICAL ERROR: The directory '{image_folder}' appears empty inside Docker!")
    exit()

# Shuffle randomly so dataset types are distributed evenly
random.seed(42)
random.shuffle(images)

# Compute 80% train and 20% validation splits
split_idx = int(len(images) * 0.8)
train_images = images[:split_idx]
val_images = images[split_idx:]

def partition_data(image_list, split_name):
    for img_name in image_list:
        base_name, _ = os.path.splitext(img_name)
        lbl_name = base_name + ".txt"
        
        src_img = os.path.join(image_folder, img_name)
        src_lbl = os.path.join(label_folder, lbl_name)
        
        dest_img = os.path.join(dest_dir, split_name, "images", img_name)
        dest_lbl = os.path.join(dest_dir, split_name, "labels", lbl_name)
        
        # Copy image file
        shutil.copy(src_img, dest_img)
            
        # Copy label or generate blank sheets for background negatives
        if os.path.exists(src_lbl):
            shutil.copy(src_lbl, dest_lbl)
        else:
            open(dest_lbl, 'w').close()

partition_data(train_images, "train")
partition_data(val_images, "val")

print(f"Split complete! Successfully processed: {len(images)} images.")
print(f"Allocated {len(train_images)} to train/ split and {len(val_images)} to val/ split.")