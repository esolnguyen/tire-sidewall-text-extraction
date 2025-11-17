import cv2
import numpy as np

def detect_tire_and_rim(yolo_model, image):
    results = yolo_model(image)
    if results and results[0].boxes:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        wheel_contour = boxes[0]  # Assuming the first box is the tire
        rim_contour = boxes[1] if len(boxes) > 1 else None  # Assuming the second box is the rim
        return wheel_contour, rim_contour
    return None

def crop_tire(image, wheel_contour, rim_contour):
    x1, y1, x2, y2 = map(int, wheel_contour)
    cropped_tire = image[y1:y2, x1:x2]
    return cropped_tire

def crop_rim(image, rim_contour):
    if rim_contour is not None:
        x1, y1, x2, y2 = map(int, rim_contour)
        cropped_rim = image[y1:y2, x1:x2]
        return cropped_rim
    return None