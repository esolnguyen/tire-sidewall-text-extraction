# Complete Tire Information Extraction Pipeline

## 🎯 What This Does

This pipeline takes a tire image (from file or URL) and extracts:

- **Manufacturer**: e.g., "Michelin"
- **Model**: e.g., "Pilot Sport 4S"
- **Size**: e.g., "245/35ZR20"
- **Load/Speed**: e.g., "95Y"
- **DOT**: e.g., "1023" (week 10, year 2023)
- **Special Markings**: e.g., ["XL", "AO"]

## 🚀 Quick Start (3 Steps)

### 1. Install Dependencies

```bash
pip install google-genai ultralytics opencv-python numpy pillow torch torchvision requests
```

### 2. Set API Key

```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

### 3. Run Pipeline

```bash
cd src
python main.py /path/to/tire_image.jpg
```

## 📋 Detailed Usage

### Command Line

```bash
# Basic usage
python main.py tire.jpg

# From URL
python main.py https://example.com/tire.jpg

# Save to JSON
python main.py tire.jpg --output results.json

# Debug mode (saves intermediate images)
python main.py tire.jpg --debug

# Verbose output
python main.py tire.jpg --verbose
```

### Python Code

```python
from src.pipeline import TireImageProcessingPipeline

# Initialize pipeline
pipeline = TireImageProcessingPipeline()

# Process image (URL or local path)
tire_info = pipeline.run_pipeline("tire.jpg")

# Access results
print(f"Brand: {tire_info.manufacturer}")
print(f"Model: {tire_info.model}")
print(f"Size: {tire_info.size}")
print(f"DOT: {tire_info.dot}")

# Get as dictionary
data = tire_info.to_dict()
```

## 📁 Project Structure

```
thesis/
├── src/
│   ├── main.py                    # CLI entry point
│   ├── pipeline.py                # Main pipeline orchestration
│   ├── config.py                  # Configuration
│   ├── models/
│   │   ├── text_detection.py     # Text detection model
│   │   └── text_recognition.py   # Text recognition model
│   ├── services/
│   │   └── gemini_service.py     # LLM extraction
│   └── types/
│       └── tire_info.py           # Data structures
├── models/
│   ├── yolo.pt                    # Tire/rim detection
│   ├── text_det.pt                # Text detection
│   └── text_rec.pth               # Text recognition
├── utils/
│   └── tire_cropping.py           # Flattening utilities
├── CLIP4STR/                      # Text recognition framework
├── requirements.txt               # Python dependencies
├── QUICKSTART.md                  # This file
└── IMPLEMENTATION_SUMMARY.md      # Technical details
```

## 🔧 How It Works

1. **Load Image**: From file or URL
2. **Detect Tire**: YOLO finds tire and rim contours
3. **Flatten Sidewall**: Unwrap to 6400x640 image
4. **Preprocess**: Apply CLAHE contrast enhancement
5. **Detect Text**: YOLO finds text regions
6. **Recognize Text**: CLIP4STR reads the text
7. **Extract Info**: Gemini LLM structures the data

## 📊 Example Output

```json
{
  "manufacturer": "Michelin",
  "model": "Pilot Sport 4S",
  "size": "245/35ZR20",
  "load_speed": "95Y",
  "dot": "1023",
  "special_markings": ["XL", "AO"]
}
```

## ⚙️ Configuration

Edit `src/config.py` to customize:

```python
class TireExtractionConfig:
    # Model paths
    YOLO_MODEL_PATH = "models/yolo.pt"
    TEXT_DETECTION_MODEL_PATH = "models/text_det.pt"
    TEXT_RECOGNITION_MODEL_PATH = "models/text_rec.pth"

    # Image dimensions
    TARGET_WIDTH = 6400
    TARGET_HEIGHT = 640

    # Gemini settings
    GEMINI_MODEL = "gemini-2.0-flash-exp"

    # Device
    DEVICE = "cpu"  # or "cuda"
```

## 🐛 Troubleshooting

### Issue: "No tire detected"

**Solution**: Ensure tire is clearly visible. Try different lighting/angle.

### Issue: "No text detected"

**Solution**: Run with `--debug` to see flattened image. Check if text is visible.

### Issue: "GEMINI_API_KEY not configured"

**Solution**:

```bash
export GEMINI_API_KEY="your-key"
# or create .env file with: GEMINI_API_KEY=your-key
```

### Issue: Import errors

**Solution**:

```bash
pip install -r requirements.txt
```

### Issue: Model not found

**Solution**: Verify model files exist in `models/` directory:

```bash
ls -lh models/
```

## 🔍 Debug Mode

Use `--debug` to save intermediate images:

```bash
python main.py tire.jpg --debug
```

This creates:

- `debug_flattened.jpg` - Flattened sidewall
- `debug_preprocessed.jpg` - After CLAHE

Inspect these to troubleshoot detection issues.

## 📦 Batch Processing

Process multiple images:

```python
from src.pipeline import TireImageProcessingPipeline

pipeline = TireImageProcessingPipeline()

images = ["tire1.jpg", "tire2.jpg", "tire3.jpg"]
results = []

for img in images:
    try:
        info = pipeline.run_pipeline(img)
        results.append(info.to_dict())
    except Exception as e:
        print(f"Failed {img}: {e}")
```

## 🌐 URL Support

Process images from the web:

```bash
python main.py https://example.com/tire.jpg
```

The pipeline automatically:

- Downloads the image
- Processes it
- Returns structured data

## 💡 Tips

1. **Better Images = Better Results**: Clear, well-lit tire images work best
2. **Debug Mode**: Always use `--debug` first to verify pipeline steps
3. **API Key**: Store in `.env` file for convenience
4. **Batch Mode**: Reuse pipeline object for multiple images (faster)
5. **Custom Config**: Adjust flattening params if text not detected

## 📚 Documentation

- `src/README.md` - Detailed API documentation
- `IMPLEMENTATION_SUMMARY.md` - Architecture and design
- `examples.py` - Code examples

## 🎓 Example Session

```bash
# 1. Set API key
export GEMINI_API_KEY="your-key"

# 2. Test with debug
cd src
python main.py ../data/test/tire1.jpg --debug --verbose

# 3. Check debug images
open debug_flattened.jpg
open debug_preprocessed.jpg

# 4. If good, run without debug
python main.py ../data/test/tire1.jpg --output tire1_info.json

# 5. View results
cat tire1_info.json
```

## 🆘 Getting Help

1. Run with `--verbose --debug` to see everything
2. Check debug images for visual issues
3. Verify all 3 model files exist
4. Test API key: `echo $GEMINI_API_KEY`
5. Review logs for specific errors

## ✅ Requirements Checklist

- [ ] Python 3.8+
- [ ] All dependencies installed (`requirements.txt`)
- [ ] Gemini API key set
- [ ] Model files in `models/` directory
- [ ] Test image available

## 🔗 Related Files

- **Main Pipeline**: `src/pipeline.py`
- **CLI Interface**: `src/main.py`
- **Configuration**: `src/config.py`
- **Examples**: `examples.py`

---

**Ready to extract tire information! 🚗💨**
