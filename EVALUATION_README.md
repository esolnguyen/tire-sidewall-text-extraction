# Tire Information Extraction - Evaluation Guide

This document explains how to evaluate the tire extraction pipeline against ground truth data.

## Data Structure

The evaluation system expects the following structure:

```
data/
├── images/           # Tire images
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── tires.csv        # Ground truth data
```

## CSV Format

The `tires.csv` file should have the following columns:

- **Manufacturer**: Tire manufacturer name (e.g., "MICHELIN", "Goodyear")
- **Model**: Tire model name (e.g., "PRIMACY 4", "EfficientGrip Performance")
- **LoadSpeed**: Load index and speed rating (e.g., "103V", "91Y")
- **Size**: Tire size specification (e.g., "225/60 R 17", "235/45R21")
- **DOT**: DOT date code (4 digits WWYY, e.g., "2021", "119")
- **SpecialMarkings**: Special markings (e.g., "XL", "M+S,3PMSF", "AO")
- **FileName**: Image filename (e.g., "img_001.jpg")

## Running Evaluation

### Basic Usage

Process all images and generate a report:

```bash
python evaluate.py
```

This will:

- Process all images in `data/images/`
- Compare results against `data/tires.csv`
- Generate `evaluation_results.json` with detailed results
- Generate `evaluation_results_summary.csv` with per-image summary
- Print a summary report to console

### Advanced Options

#### Process specific number of images (for testing)

```bash
python evaluate.py --max-images 10
```

#### Start from specific image index

```bash
python evaluate.py --start-from 50 --max-images 20
```

This processes images 50-69.

#### Save debug images

```bash
python evaluate.py --debug
```

This saves intermediate processing images to `debug/` folder for each tire.

#### Custom output file

```bash
python evaluate.py --output my_results.json
```

#### Custom data directory

```bash
python evaluate.py --data-dir /path/to/data
```

### Combined Example

```bash
python evaluate.py --max-images 20 --debug --output test_run.json
```

## Output Files

### 1. JSON Results (`evaluation_results.json`)

Detailed results including:

- Summary statistics
- Field-level accuracy for each field (Manufacturer, Model, Size, etc.)
- Per-image detailed results
- Error messages and execution times

Example structure:

```json
{
  "timestamp": "2025-11-16T10:30:00",
  "summary": {
    "total_images": 100,
    "successful": 95,
    "failed": 5,
    "overall_correct": 60,
    "overall_accuracy": 63.16,
    "avg_execution_time": 8.5
  },
  "field_statistics": {
    "Manufacturer": {
      "correct": 85,
      "partial": 5,
      "incorrect": 5,
      "accuracy": 89.47
    },
    ...
  },
  "detailed_results": [...]
}
```

### 2. CSV Summary (`evaluation_results_summary.csv`)

Per-image summary with:

- filename
- success (True/False)
- overall_correct (all fields correct)
- execution_time
- Individual field correctness (manufacturer_correct, model_correct, etc.)
- error message (if failed)

## Evaluation Metrics

### Field-Level Accuracy

Each field is evaluated independently:

- **Correct**: Exact match (after normalization)
- **Partial**: Substring match or partial overlap
- **Incorrect**: No match

For `SpecialMarkings`, Jaccard similarity is calculated for partial matches.

### Overall Accuracy

Percentage of images where ALL fields are correct.

### Execution Time

Average time to process one image through the complete pipeline.

## Console Output

The script provides real-time feedback:

```
[1/100] Processing img_001.jpg
================================================================================
Processing: img_001.jpg
================================================================================
Expected: {'Manufacturer': 'GOODYEAR', 'Model': 'EfficientGrip Performance', ...}
Predicted: {'Manufacturer': 'GOODYEAR', 'Model': 'EfficientGrip Performance', ...}
  ✓ Manufacturer: Exact match
  ✓ Model: Exact match
  ✓ LoadSpeed: Exact match
  ✓ Size: Exact match
  ✓ DOT: Exact match
  ✓ SpecialMarkings: Exact match
✓✓✓ ALL FIELDS CORRECT ✓✓✓
```

Symbols:

- ✓ = Correct
- ~ = Partial match
- ✗ = Incorrect

## Tips

1. **Start small**: Use `--max-images 5` to test the evaluation script first
2. **Debug mode**: Use `--debug` to save intermediate images for failed cases
3. **Incremental evaluation**: Use `--start-from` to continue evaluation after interruption
4. **Monitor progress**: The script shows detailed progress for each image
5. **Check CSV format**: Ensure your CSV has the correct column names and format

## Troubleshooting

### Import Errors

Make sure you're running from the project root directory:

```bash
cd /Users/nguyenthang/Workspace/thesis
python evaluate.py
```

### Missing API Key

Ensure `GEMINI_API_KEY` is set in your `.env` file:

```bash
echo "GEMINI_API_KEY=your_key_here" > .env
```

### HEIC Images

Some images are in HEIC format. Make sure you have the necessary libraries:

```bash
pip install pillow-heif
```

### Memory Issues

If processing many images causes memory issues, process in batches:

```bash
python evaluate.py --start-from 0 --max-images 50
python evaluate.py --start-from 50 --max-images 50
python evaluate.py --start-from 100 --max-images 50
```

Then combine the results manually.

## Understanding Results

### High Accuracy (>90%)

Field is being extracted reliably. Minor issues may be OCR-related.

### Medium Accuracy (70-90%)

Field extraction is working but has some issues. Check:

- OCR quality
- Prompt engineering
- Model capability

### Low Accuracy (<70%)

Significant issues with field extraction. Investigate:

- Image quality and preprocessing
- Detection model performance
- LLM prompt effectiveness
- Ground truth data quality

## Next Steps

After evaluation:

1. **Review failed cases**: Look at debug images to understand failures
2. **Analyze error patterns**: Check if certain manufacturers/models fail consistently
3. **Improve pipeline**: Based on results, improve:
   - Image preprocessing
   - Text detection
   - Text recognition
   - LLM prompts
4. **Re-evaluate**: Run evaluation again after improvements
