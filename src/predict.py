import os
from ultralytics import YOLO

def main():
    # 1. Load your trained model weights
    model = YOLO("/home/Brandon/projects/scrap-metal-ai/runs/detect/scrap_metal_detector_150e/weights/best.pt")

    # 2. Run prediction on an image or folder of images
    # Replace this path with where your new test images are stored
    input_source = "/home/Brandon/projects/scrap-metal-ai/testimages" 

    results = model.predict(
        source=input_source,
        conf=0.30,       # Only show boxes with over 30% confidence
        save=True,       # Save the images with bounding boxes drawn on them
        line_width=2,    # Adjust bounding box outline thickness
        show_labels=True, # Ensure text labels are printed on the box
        show_conf=True   # Show the percentage confidence next to the label name
    )

    print("Prediction complete! Check your 'runs/detect/predict/' directory.")

if __name__ == "__main__":
    main()