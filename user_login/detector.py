# user_login/detector.py

from ultralytics import YOLO
import os
from django.conf import settings

# Use absolute path for the model file
model_path = os.path.join(settings.BASE_DIR, "user_login", "static", "best.pt")

# Load the model
model = YOLO(model_path)

def detect_from_image(image_path):
    """
    Detect objects from an image file path.
    Saves results to MEDIA_ROOT/detected/
    Returns the relative path to the saved image.
    """
    save_dir = os.path.join(settings.MEDIA_ROOT, "detected")
    os.makedirs(save_dir, exist_ok=True)

    results = model(image_path)
    for r in results:
        r.save(save_dir=save_dir)  # Save outputs here

    # Return relative path to saved image for serving in Django
    filename = os.path.basename(image_path)
    return f"detected/{filename}"

def detect_from_video(video_path):
    """
    Detect objects from a video file path.
    Saves results to default YOLO runs/detect/predict directory.
    Returns that directory path.
    """
    results = model.predict(source=video_path, save=True, save_txt=False)
    # The default save location can be configured or hardcoded:
    return "runs/detect/predict/"

def detect_from_webcam():
    """
    Live detect from webcam (source=0).
    Shows window with detections.
    """
    model.predict(source=0, show=True)
