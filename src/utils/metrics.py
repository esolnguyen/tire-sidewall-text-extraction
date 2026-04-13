import os
import csv
import cv2
import numpy as np
from scipy.stats import entropy as shannon_entropy


METRICS_COLUMNS = [
    "image",
    "method",
    "entropy",
    "std_dev",
    "michelson_contrast",
    "mean_brightness",
    "histogram_flatness",
]


def compute_metrics(img: np.ndarray) -> dict:
    """Compute contrast/quality metrics for a grayscale image."""
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / hist.sum()
    ent = shannon_entropy(hist_norm, base=2)
    std = np.std(img.astype(np.float64))
    i_max, i_min = float(np.max(img)), float(np.min(img))
    michelson = (i_max - i_min) / (i_max + i_min) if (i_max + i_min) > 0 else 0
    mean = np.mean(img.astype(np.float64))
    hist_nz = hist_norm[hist_norm > 0]
    flatness = (
        np.exp(np.mean(np.log(hist_nz))) /
        np.mean(hist_nz) if len(hist_nz) > 0 else 0
    )
    return {
        "entropy": round(ent, 4),
        "std_dev": round(std, 2),
        "michelson_contrast": round(michelson, 4),
        "mean_brightness": round(mean, 2),
        "histogram_flatness": round(flatness, 4),
    }


def append_metrics_to_csv(
    csv_path: str, image_name: str, method: str, metrics: dict
):
    """Append one row of metrics to the CSV file."""
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({"image": image_name, "method": method, **metrics})
