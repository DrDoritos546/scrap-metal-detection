import os
from PIL import Image

dataset_path = "/workspace/scrap_dataset"

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(root, file)
            try:
                # Force open the image and verify its integrity
                with Image.open(img_path) as img:
                    img.verify()
                
                # Re-open to resave cleanly as standard RGB, stripping any bad headers
                with Image.open(img_path) as img:
                    clean_img = img.convert('RGB')
                    clean_img.save(img_path, 'JPEG', quality=95)
                    
            except Exception as e:
                print(f"❌ CORRUPTED IMAGE FOUND AND REMOVED: {img_path}")
                print(f"Reason: {e}")
                # Delete the corrupt image
                if os.path.exists(img_path):
                    os.remove(img_path)
                # Also delete its matching .txt label file if it exists
                label_path = img_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
                if os.path.exists(label_path):
                    os.remove(label_path)