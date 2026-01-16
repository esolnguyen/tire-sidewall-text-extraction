#!/usr/bin/env python3
"""
Preprocessing Methods Comparison Script

This script compares three preprocessing approaches for the thesis:
1. Linear Stretching [14]
2. Histogram Equalization (HE) [15]
3. Contrast-Limited Adaptive Histogram Equalization (CLAHE) [16]

Usage:
    python compare_preprocessing.py --image path/to/image.jpg
    python compare_preprocessing.py --evaluate-dataset
"""

from src.pipeline import TireImageProcessingPipeline
from src.config import TireExtractionConfig
from src.utils.image_preprocessing import (
    linear_stretching,
    histogram_equalization,
    clahe_enhancement,
    preprocess_image
)
import argparse
import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from typing import Dict, List
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))


def visualize_preprocessing_comparison(image_path: str, save_dir: str = "preprocessing_comparison"):
    """Compare all three preprocessing methods visually

    Args:
        image_path: Path to input image
        save_dir: Directory to save comparison results
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Apply each method
    original = image.copy()
    linear = linear_stretching(image)
    hist_eq = histogram_equalization(image)
    clahe = clahe_enhancement(image)

    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    methods = [
        (original, "Original", axes[0, 0]),
        (linear, "Linear Stretching", axes[0, 1]),
        (hist_eq, "Histogram Equalization", axes[1, 0]),
        (clahe, "CLAHE", axes[1, 1])
    ]

    for img, title, ax in methods:
        # Convert BGR to RGB for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()

    # Save comparison
    os.makedirs(save_dir, exist_ok=True)
    comparison_path = os.path.join(save_dir, "preprocessing_comparison.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison to: {comparison_path}")

    # Save individual processed images
    base_name = Path(image_path).stem
    cv2.imwrite(os.path.join(save_dir, f"{base_name}_original.jpg"), original)
    cv2.imwrite(os.path.join(save_dir, f"{base_name}_linear.jpg"), linear)
    cv2.imwrite(os.path.join(
        save_dir, f"{base_name}_histogram_eq.jpg"), hist_eq)
    cv2.imwrite(os.path.join(save_dir, f"{base_name}_clahe.jpg"), clahe)
    print(f"✓ Saved individual images to: {save_dir}/")

    # Show histogram comparison
    visualize_histograms(original, linear, hist_eq, clahe, save_dir, base_name)

    plt.show()


def visualize_histograms(original, linear, hist_eq, clahe, save_dir, base_name):
    """Compare histograms of different preprocessing methods"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    methods = [
        (original, "Original", axes[0, 0]),
        (linear, "Linear Stretching", axes[0, 1]),
        (hist_eq, "Histogram Equalization", axes[1, 0]),
        (clahe, "CLAHE", axes[1, 1])
    ]

    for img, title, ax in methods:
        # Convert to grayscale for histogram
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

        # Plot
        ax.plot(hist, color='black')
        ax.set_title(f"{title} - Histogram", fontsize=12, fontweight='bold')
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')
        ax.set_xlim([0, 256])
        ax.grid(alpha=0.3)

        # Add statistics
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        ax.text(0.98, 0.97, f'μ={mean_val:.1f}\nσ={std_val:.1f}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    hist_path = os.path.join(save_dir, f"{base_name}_histograms.png")
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved histogram comparison to: {hist_path}")


def evaluate_preprocessing_methods(data_csv: str = "data/tires.csv",
                                   output_file: str = "preprocessing_evaluation.json"):
    """Evaluate all three preprocessing methods on the dataset

    Args:
        data_csv: Path to dataset CSV
        output_file: Path to save evaluation results
    """
    import pandas as pd

    methods = ['linear', 'histogram_eq', 'clahe', 'none']
    results = {}

    print("=" * 80)
    print("PREPROCESSING METHODS EVALUATION")
    print("=" * 80)

    for method in methods:
        print(f"\nEvaluating method: {method.upper()}")
        print("-" * 80)

        # Create config with this method
        config = TireExtractionConfig()
        config.PREPROCESSING_METHOD = method

        # Initialize pipeline
        pipeline = TireImageProcessingPipeline(config)

        # Load dataset
        df = pd.read_csv(data_csv)

        # Run evaluation on subset (first 50 images for quick comparison)
        n_samples = min(50, len(df))
        correct = 0
        total = 0
        errors = []

        for idx, row in df.head(n_samples).iterrows():
            filename = row['filename']
            image_path = os.path.join("data/images", filename)

            if not os.path.exists(image_path):
                continue

            try:
                # Run pipeline
                tire_info = pipeline.run_pipeline(image_path, save_debug=False)

                # Simple check: did we extract something?
                if tire_info.manufacturer or tire_info.model or tire_info.size:
                    correct += 1

                total += 1

                if total % 10 == 0:
                    print(f"  Processed {total}/{n_samples} images...")

            except Exception as e:
                errors.append(str(e))
                total += 1

        # Calculate metrics
        accuracy = (correct / total * 100) if total > 0 else 0

        results[method] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'errors': len(errors)
        }

        print(f"\n  Results for {method}:")
        print(f"    Accuracy: {accuracy:.2f}%")
        print(f"    Correct: {correct}/{total}")
        print(f"    Errors: {len(errors)}")

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nMethod Comparison:")
    for method, metrics in results.items():
        print(
            f"  {method:15s}: {metrics['accuracy']:6.2f}% ({metrics['correct']}/{metrics['total']})")

    print(f"\n✓ Results saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare preprocessing methods for tire text extraction"
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path to a single image for visual comparison'
    )
    parser.add_argument(
        '--evaluate-dataset',
        action='store_true',
        help='Evaluate all methods on the full dataset'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='preprocessing_comparison',
        help='Output directory for results'
    )

    args = parser.parse_args()

    if args.image:
        # Visual comparison of single image
        visualize_preprocessing_comparison(args.image, args.output_dir)
    elif args.evaluate_dataset:
        # Full dataset evaluation
        evaluate_preprocessing_methods()
    else:
        # Default: show help
        parser.print_help()
        print("\nExample usage:")
        print("  python compare_preprocessing.py --image data/images/img_001.jpg")
        print("  python compare_preprocessing.py --evaluate-dataset")


if __name__ == "__main__":
    main()
