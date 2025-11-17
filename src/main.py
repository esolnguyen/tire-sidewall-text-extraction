#!/usr/bin/env python3
"""
Tire Information Extraction System

Main entry point for extracting tire information from images.
Supports both local file paths and URLs.

Usage:
    python main.py <image_path_or_url> [options]
    
Examples:
    python main.py data/tire.jpg
    python main.py https://example.com/tire.jpg --debug
    python main.py tire.jpg --output results.json
"""

import os
import sys
import json
import logging
import argparse
from typing import Optional
from pipeline import TireImageProcessingPipeline
from config import TireExtractionConfig
from models_types.tire_info import TireInfo

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_tire_image(
    image_source: str,
    output_file: Optional[str] = None,
    save_debug: bool = False
) -> TireInfo:
    """Process a tire image and extract information

    Args:
        image_source: Path to image file or URL
        output_file: Optional path to save JSON output
        save_debug: Whether to save debug images

    Returns:
        TireInfo object with extracted information
    """
    # Initialize pipeline
    config = TireExtractionConfig()
    pipeline = TireImageProcessingPipeline(config)

    # Run the pipeline
    tire_info = pipeline.run_pipeline(image_source, save_debug=save_debug)

    # Save to file if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(tire_info.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_file}")

    return tire_info


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Extract tire information from images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s tire.jpg
  %(prog)s https://example.com/tire.jpg --debug
  %(prog)s tire.jpg --output results.json
  %(prog)s tire.jpg --output results.json --debug
        """
    )

    parser.add_argument(
        'image',
        help='Path to tire image file or URL'
    )

    parser.add_argument(
        '--output', '-o',
        help='Output JSON file path',
        default=None
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Save intermediate debug images'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        logger.info(f"Processing image: {args.image}")

        # Process the image
        tire_info = process_tire_image(
            args.image,
            output_file=args.output,
            save_debug=args.debug
        )

        # Print results
        print("\n" + "="*60)
        print("TIRE INFORMATION EXTRACTION RESULTS")
        print("="*60)
        print(tire_info)
        print("="*60 + "\n")

        # Also print as JSON
        if not args.output:
            print("JSON Output:")
            print(json.dumps(tire_info.to_dict(), indent=2, ensure_ascii=False))

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return 1
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
