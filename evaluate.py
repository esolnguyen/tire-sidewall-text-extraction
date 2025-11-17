"""
Tire Information Extraction - Evaluation Script

This script evaluates the tire extraction pipeline against ground truth data.
It processes all images in the data/images folder and compares the results
against the expected values in data/tires.csv.

Usage:
    python evaluate.py [options]
    
Examples:
    python evaluate.py
    python evaluate.py --output evaluation_results.json
    python evaluate.py --debug --max-images 10
"""

from src.config import TireExtractionConfig
from src.pipeline import TireImageProcessingPipeline
import os
import sys
import csv
import json
import logging
import argparse
import pandas as pd
from datetime import datetime
from typing import Dict, List,  Optional
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normalize text for comparison - handles NaN, empty, and special characters"""
    import math

    # Handle None, NaN, empty string, "Not found", "Not available"
    if text is None:
        return ""
    if isinstance(text, float) and math.isnan(text):
        return ""

    text_str = str(text).strip()

    # Handle empty or placeholder strings
    if text_str in ["", "Not found", "Not available", "nan", "NaN", "NULL", "null"]:
        return ""

    # Convert to uppercase and remove spaces
    text_str = text_str.upper().replace(" ", "")

    # Replace special characters with numeric equivalents
    # Common superscript/subscript numbers
    superscript_map = {
        '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
        '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
        '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
        '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9'
    }

    for special, numeric in superscript_map.items():
        text_str = text_str.replace(special, numeric)

    return text_str


def normalize_special_markings(markings: any) -> set:
    """Normalize special markings to a set for comparison - handles NaN and empty values"""
    import math

    # Handle None and NaN
    if markings is None:
        return set()
    if isinstance(markings, float) and math.isnan(markings):
        return set()
    if not markings:
        return set()

    if isinstance(markings, str):
        # Handle empty or placeholder strings
        if markings.strip() in ["Not found", "Not available", "", "[]", "nan", "NaN", "NULL", "null"]:
            return set()
        # Handle string representation of list
        if markings.startswith('[') and markings.endswith(']'):
            markings = markings[1:-1]
        # Split by comma and normalize each item
        items = [item.strip().strip('"').strip("'")
                 for item in markings.split(',')]
        result = set()
        for item in items:
            normalized = normalize_text(item)
            if normalized:  # Only add non-empty normalized items
                result.add(normalized)
        return result

    if isinstance(markings, list):
        result = set()
        for item in markings:
            normalized = normalize_text(str(item))
            if normalized:  # Only add non-empty normalized items
                result.add(normalized)
        return result

    # Single value
    normalized = normalize_text(str(markings))
    return {normalized} if normalized else set()


def calculate_field_accuracy(predicted: str, expected: str, field_name: str) -> Dict:
    """Calculate accuracy for a single field"""
    pred_norm = normalize_text(predicted)
    exp_norm = normalize_text(expected)

    # Handle special markings separately
    if field_name == "SpecialMarkings":
        pred_set = normalize_special_markings(predicted)
        exp_set = normalize_special_markings(expected)

        if not exp_set and not pred_set:
            return {"correct": True, "partial": False, "details": "Both empty"}
        if not exp_set:
            return {"correct": False, "partial": False, "details": f"Expected empty but got {pred_set}"}
        if not pred_set:
            return {"correct": False, "partial": False, "details": f"Got empty but expected {exp_set}"}

        # Calculate Jaccard similarity
        intersection = pred_set & exp_set
        union = pred_set | exp_set

        if pred_set == exp_set:
            return {"correct": True, "partial": False, "details": "Exact match"}
        elif len(intersection) > 0:
            similarity = len(intersection) / len(union)
            return {
                "correct": False,
                "partial": True,
                "similarity": similarity,
                "details": f"Partial match: {intersection} (missing: {exp_set - pred_set}, extra: {pred_set - exp_set})"
            }
        else:
            return {
                "correct": False,
                "partial": False,
                "details": f"No match. Expected: {exp_set}, Got: {pred_set}"
            }

    # Handle other fields
    if not exp_norm and not pred_norm:
        return {"correct": True, "partial": False, "details": "Both empty"}
    if not exp_norm:
        return {"correct": False, "partial": False, "details": f"Expected empty but got '{predicted}'"}
    if not pred_norm:
        return {"correct": False, "partial": False, "details": f"Got empty but expected '{expected}'"}

    if pred_norm == exp_norm:
        return {"correct": True, "partial": False, "details": "Exact match"}

    # Check for partial match (substring)
    if pred_norm in exp_norm or exp_norm in pred_norm:
        return {
            "correct": False,
            "partial": True,
            "details": f"Partial match. Expected: '{expected}', Got: '{predicted}'"
        }

    return {
        "correct": False,
        "partial": False,
        "details": f"No match. Expected: '{expected}', Got: '{predicted}'"
    }


def evaluate_single_image(
    image_path: str,
    expected: Dict,
    pipeline: TireImageProcessingPipeline,
    save_debug: bool = False
) -> Dict:
    """Evaluate a single image"""
    filename = os.path.basename(image_path)
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing: {filename}")
    logger.info(f"{'='*80}")

    result = {
        "filename": filename,
        "success": False,
        "error": None,
        "predicted": {},
        "expected": expected,
        "field_results": {},
        "overall_correct": False,
        "execution_time": 0
    }

    try:
        import time
        start_time = time.time()

        # Run pipeline
        tire_info = pipeline.run_pipeline(image_path, save_debug=save_debug)

        result["execution_time"] = time.time() - start_time
        result["success"] = True

        # Extract predicted values
        result["predicted"] = {
            "Manufacturer": tire_info.manufacturer or "Not found",
            "Model": tire_info.model or "Not found",
            "LoadSpeed": tire_info.load_speed or "Not found",
            "Size": tire_info.size or "Not found",
            "DOT": tire_info.dot or "Not found",
            "SpecialMarkings": tire_info.special_markings or []
        }

        logger.info(f"Expected: {expected}")
        logger.info(f"Predicted: {result['predicted']}")

        # Calculate field-level accuracy
        fields = ["Manufacturer", "Model", "LoadSpeed",
                  "Size", "DOT", "SpecialMarkings"]
        all_correct = True

        for field in fields:
            field_result = calculate_field_accuracy(
                result["predicted"].get(field, ""),
                expected.get(field, ""),
                field
            )
            result["field_results"][field] = field_result

            if not field_result["correct"]:
                all_correct = False

            status = "✓" if field_result["correct"] else (
                "~" if field_result.get("partial", False) else "✗")
            logger.info(f"  {status} {field}: {field_result['details']}")

        result["overall_correct"] = all_correct

        if all_correct:
            logger.info(f"✓✓✓ ALL FIELDS CORRECT ✓✓✓")

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        logger.error(f"Error processing {filename}: {e}")
        logger.debug(traceback.format_exc())

    return result


def generate_report(results: List[Dict], output_file: Optional[str] = None) -> Dict:
    """Generate evaluation report"""
    total_images = len(results)
    successful = sum(1 for r in results if r["success"])
    failed = total_images - successful

    # Calculate field-level accuracies
    fields = ["Manufacturer", "Model", "LoadSpeed",
              "Size", "DOT", "SpecialMarkings"]
    field_stats = {}

    for field in fields:
        correct = 0
        partial = 0
        incorrect = 0
        total_similarity = 0
        count_with_similarity = 0

        for result in results:
            if result["success"] and field in result["field_results"]:
                fr = result["field_results"][field]
                if fr["correct"]:
                    correct += 1
                elif fr.get("partial", False):
                    partial += 1
                    if "similarity" in fr:
                        total_similarity += fr["similarity"]
                        count_with_similarity += 1
                else:
                    incorrect += 1

        total_processed = correct + partial + incorrect
        accuracy = (correct / total_processed *
                    100) if total_processed > 0 else 0
        partial_rate = (partial / total_processed *
                        100) if total_processed > 0 else 0
        avg_similarity = (
            total_similarity / count_with_similarity) if count_with_similarity > 0 else 0

        field_stats[field] = {
            "correct": correct,
            "partial": partial,
            "incorrect": incorrect,
            "total": total_processed,
            "accuracy": accuracy,
            "partial_rate": partial_rate,
            "avg_similarity": avg_similarity
        }

    # Overall accuracy
    overall_correct = sum(
        1 for r in results if r.get("overall_correct", False))
    overall_accuracy = (overall_correct / successful *
                        100) if successful > 0 else 0

    # Average execution time
    avg_time = sum(r["execution_time"] for r in results if r["success"]
                   ) / successful if successful > 0 else 0

    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_images": total_images,
            "successful": successful,
            "failed": failed,
            "overall_correct": overall_correct,
            "overall_accuracy": overall_accuracy,
            "avg_execution_time": avg_time
        },
        "field_statistics": field_stats,
        "detailed_results": results
    }

    # Print report
    print("\n" + "="*80)
    print("EVALUATION REPORT")
    print("="*80)
    print(f"Total Images: {total_images}")
    print(
        f"Successfully Processed: {successful} ({successful/total_images*100:.1f}%)")
    print(f"Failed: {failed} ({failed/total_images*100:.1f}%)")
    print(
        f"Overall Correct (All Fields): {overall_correct} ({overall_accuracy:.1f}%)")
    print(f"Average Execution Time: {avg_time:.2f}s")
    print("\n" + "-"*80)
    print("FIELD-LEVEL ACCURACY")
    print("-"*80)

    for field, stats in field_stats.items():
        print(f"\n{field}:")
        print(f"  Correct:   {stats['correct']:4d} ({stats['accuracy']:.1f}%)")
        print(
            f"  Partial:   {stats['partial']:4d} ({stats['partial_rate']:.1f}%)")
        print(f"  Incorrect: {stats['incorrect']:4d}")
        if stats['avg_similarity'] > 0:
            print(
                f"  Avg Similarity (for partial): {stats['avg_similarity']:.2f}")

    print("\n" + "="*80)

    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nDetailed results saved to: {output_file}")

        # Also save CSV summary
        csv_file = output_file.replace('.json', '_summary.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'filename', 'success', 'overall_correct', 'execution_time',
                'manufacturer_correct', 'model_correct', 'loadspeed_correct',
                'size_correct', 'dot_correct', 'specialmarkings_correct', 'error'
            ])
            writer.writeheader()
            for r in results:
                row = {
                    'filename': r['filename'],
                    'success': r['success'],
                    'overall_correct': r.get('overall_correct', False),
                    'execution_time': f"{r['execution_time']:.2f}",
                    'error': r.get('error', '')
                }
                if r['success']:
                    for field in ['Manufacturer', 'Model', 'LoadSpeed', 'Size', 'DOT', 'SpecialMarkings']:
                        key = f"{field.lower()}_correct"
                        row[key] = r['field_results'][field]['correct']
                writer.writerow(row)
        print(f"Summary CSV saved to: {csv_file}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate tire extraction pipeline against ground truth"
    )
    parser.add_argument(
        '--data-dir',
        default='data',
        help='Directory containing images/ folder and tires.csv'
    )
    parser.add_argument(
        '--output',
        default='evaluation_results.json',
        help='Output JSON file for detailed results'
    )
    parser.add_argument(
        '--max-images',
        type=int,
        help='Maximum number of images to process (for testing)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Save debug images for each processed tire'
    )
    parser.add_argument(
        '--start-from',
        type=int,
        default=0,
        help='Start processing from this image index'
    )

    args = parser.parse_args()

    # Paths
    images_dir = os.path.join(args.data_dir, 'images')
    csv_path = os.path.join(args.data_dir, 'tires.csv')

    if not os.path.exists(images_dir):
        logger.error(f"Images directory not found: {images_dir}")
        return

    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        return

    # Load ground truth
    logger.info(f"Loading ground truth from {csv_path}")
    df = pd.read_csv(csv_path, keep_default_na=True)
    logger.info(f"Loaded {len(df)} ground truth records")

    # Helper function to safely get value from pandas
    def safe_get_value(value):
        """Convert pandas value to string, handling NaN"""
        if pd.isna(value):
            return ""
        return str(value).strip()

    # Create lookup dictionary
    ground_truth = {}
    for _, row in df.iterrows():
        filename = row['FileName']
        ground_truth[filename] = {
            'Manufacturer': safe_get_value(row['Manufacturer']),
            'Model': safe_get_value(row['Model']),
            'LoadSpeed': safe_get_value(row['LoadSpeed']),
            'Size': safe_get_value(row['Size']),
            'DOT': safe_get_value(row['DOT']),
            'SpecialMarkings': safe_get_value(row['SpecialMarkings'])
        }

    # Initialize pipeline
    logger.info("Initializing pipeline...")
    config = TireExtractionConfig()
    pipeline = TireImageProcessingPipeline(config)
    logger.info("Pipeline initialized")

    # Get list of images to process
    image_files = sorted([f for f in os.listdir(images_dir)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.heic'))])

    # Filter to only images with ground truth
    image_files = [f for f in image_files if f in ground_truth]

    logger.info(f"Found {len(image_files)} images with ground truth")

    # Apply start_from and max_images filters
    if args.start_from > 0:
        image_files = image_files[args.start_from:]
        logger.info(f"Starting from image {args.start_from}")

    if args.max_images:
        image_files = image_files[:args.max_images]
        logger.info(f"Processing only first {args.max_images} images")

    # Process images
    results = []
    for i, filename in enumerate(image_files, 1):
        logger.info(f"\n[{i}/{len(image_files)}] Processing {filename}")

        image_path = os.path.join(images_dir, filename)
        expected = ground_truth[filename]

        result = evaluate_single_image(
            image_path,
            expected,
            pipeline,
            save_debug=args.debug
        )
        results.append(result)

    # Generate report
    report = generate_report(results, args.output)

    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
