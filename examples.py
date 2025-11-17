#!/usr/bin/env python3
"""
Example script demonstrating the tire extraction pipeline

This script shows how to use the pipeline programmatically
"""

from src.config import TireExtractionConfig
from src.pipeline import TireImageProcessingPipeline
import os
import sys
import json

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def example_process_local_image():
    """Example: Process a local image file"""
    print("Example 1: Processing a local image file")
    print("-" * 60)

    # Path to your tire image
    image_path = "IMG_3395.JPEG"  # Change this to your image path

    # Initialize the pipeline
    config = TireExtractionConfig()
    pipeline = TireImageProcessingPipeline(config)

    # Process the image
    try:
        tire_info = pipeline.run_pipeline(image_path, save_debug=True)

        print("\nExtracted Tire Information:")
        print(tire_info)

        print("\nJSON Format:")
        print(json.dumps(tire_info.to_dict(), indent=2))

    except Exception as e:
        print(f"Error processing image: {e}")


def example_process_url():
    """Example: Process an image from URL"""
    print("\nExample 2: Processing an image from URL")
    print("-" * 60)

    # URL to a tire image
    image_url = "https://example.com/tire.jpg"  # Change this to a real URL

    # Initialize the pipeline
    config = TireExtractionConfig()
    pipeline = TireImageProcessingPipeline(config)

    # Process the image
    try:
        tire_info = pipeline.run_pipeline(image_url)

        print("\nExtracted Tire Information:")
        print(tire_info)

    except Exception as e:
        print(f"Error processing URL: {e}")


def example_batch_processing():
    """Example: Process multiple images"""
    print("\nExample 3: Batch processing multiple images")
    print("-" * 60)

    # List of images to process
    image_paths = [
        "data/tire1.jpg",
        "data/tire2.jpg",
        "data/tire3.jpg",
    ]

    # Initialize the pipeline once
    config = TireExtractionConfig()
    pipeline = TireImageProcessingPipeline(config)

    results = []

    for i, image_path in enumerate(image_paths, 1):
        print(f"\nProcessing image {i}/{len(image_paths)}: {image_path}")

        try:
            tire_info = pipeline.run_pipeline(image_path)
            results.append({
                "image": image_path,
                "info": tire_info.to_dict(),
                "status": "success"
            })
            print(f"✓ Success: {tire_info.manufacturer} {tire_info.model}")

        except Exception as e:
            results.append({
                "image": image_path,
                "error": str(e),
                "status": "failed"
            })
            print(f"✗ Failed: {e}")

    # Save batch results
    output_file = "batch_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nBatch results saved to: {output_file}")


def example_custom_config():
    """Example: Using custom configuration"""
    print("\nExample 4: Custom configuration")
    print("-" * 60)

    # Create custom configuration
    config = TireExtractionConfig()

    # Customize settings
    config.TARGET_WIDTH = 3200  # Reduce width for faster processing
    config.TARGET_HEIGHT = 320
    config.CLAHE_CLIP_LIMIT = 3.0  # Less aggressive contrast enhancement

    # Initialize pipeline with custom config
    pipeline = TireImageProcessingPipeline(config)

    image_path = "data/tire.jpg"

    try:
        tire_info = pipeline.run_pipeline(image_path)
        print("Processed with custom config:")
        print(tire_info)

    except Exception as e:
        print(f"Error: {e}")


def main():
    """Run examples"""
    print("="*60)
    print("TIRE EXTRACTION PIPELINE - EXAMPLES")
    print("="*60)

    # Check for API key
    if not os.getenv("GEMINI_API_KEY"):
        print("\n⚠️  WARNING: GEMINI_API_KEY not set!")
        print("Please set your API key:")
        print("  export GEMINI_API_KEY='your-key-here'")
        print("\nContinuing anyway (will fail at LLM step)...\n")

    # Run examples
    # Uncomment the examples you want to run

    example_process_local_image()
    # example_process_url()
    # example_batch_processing()
    # example_custom_config()

    print("\n" + "="*60)
    print("To run examples, uncomment them in the main() function")
    print("="*60)


if __name__ == "__main__":
    main()
