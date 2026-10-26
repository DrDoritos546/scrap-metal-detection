from ultralytics import YOLO

def main():
    # Upgrade to the Large model for significantly stronger feature recognition
    model = YOLO("yolov8l.pt")

    # Train on CPU with higher resolution and increased capability
    results = model.train(
        data="/home/Brandon/projects/scrap-metal-ai/scrap_dataset/data.yaml",
        epochs=150,        # Kept at 150 since convergence behavior is stable
        imgsz=800,         # Increased from 640 to 800 for better small object detection
        batch=4,           # Reduced batch size to 4 because Large model + 800px requires heavy CPU RAM
        device="cpu",      # Explicitly set to run on CPU
        workers=8,         # Threads for faster data loading
        name="scrap_metal_detector_large",
        exist_ok=True      # Overwrites the directory if you rerun this configuration
    )

if __name__ == "__main__":
    main()
"""
yolo task=detect mode=train \
  model=yolov8n.pt \
  data=/home/Brandon/projects/scrap-metal-ai/scrap_dataset/data.yaml \
  epochs=150 \
  imgsz=640 \
  batch=16 \
  device=cpu \
  name=scrap_metal_detector_150e
"""