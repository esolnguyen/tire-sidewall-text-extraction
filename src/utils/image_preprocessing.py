import cv2
import numpy as np

def resize_image(image: np.ndarray, target_size: tuple) -> np.ndarray:
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

def apply_clahe(image: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return clahe.apply(gray_image)

def preprocess_image(image: np.ndarray, target_size: tuple) -> np.ndarray:
    resized_image = resize_image(image, target_size)
    enhanced_image = apply_clahe(resized_image)
    return enhanced_image

def normalize_image(image: np.ndarray) -> np.ndarray:
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)