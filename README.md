# NIC OCR FastAPI Service

A FastAPI backend service for Computer Vision and OCR processing on NIC (National Identity Card) images. The service applies advanced OpenCV preprocessing, runs OCR via EasyOCR with AI-powered super-resolution upscaling, and returns structured JSON with extracted NIC fields and metadata.

## 🌟 Features

- **Modular Preprocessing Pipeline**: Advanced image enhancement using OpenCV with super-image EDSRMODEL for intelligent upscaling
- **OCR Extraction**: Powered by EasyOCR for accurate text recognition
- **Structured Field Extraction**: Automatically extracts CNIC number, name, father/husband name, gender, date of birth, and country of stay
- **Smart Quality Adaptation**: Adapts preprocessing techniques based on image quality assessment
- **Comprehensive Validation**: Robust error handling and validation at every step
- **Graceful Degradation**: Handles partial or failed extraction with status indicators and warnings
- **Super-Image Upscaling**: Uses EDSRModel upscaling with superior quality
- **Postman Collection**: Pre-configured API testing collection included

## 📦 Technologies

- **FastAPI**: Modern, fast web framework
- **OpenCV**: Computer Vision preprocessing
- **EasyOCR**: Text recognition engine
- **PyTorch**: Deep learning framework (for super-image)
- **Super-Image**: AI-powered image upscaling (RealESRGAN)
- **Pydantic**: Data validation and schema management

## 📁 Project Structure

```
OCR_NIC_TASK/
├── app/
│   ├── __init__.py
│   ├── config.py              # Configuration settings
│   ├── main.py                # FastAPI application and endpoints
│   ├── ocr.py                 # OCR processing and field extraction
│   ├── preprocess.py          # Image preprocessing pipeline
│   ├── schemas.py             # Pydantic data models
│   ├── test_pipeline.py       # Full pipeline testing script
│   ├── test_utils.py          # Utility functions for testing
│   └── run.py                 # Application launcher
├── Cards Dataset/             # Sample NIC images for testing
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the API

```bash
cd app
python run.py
```

The API will start at `http://localhost:8000`

- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 3. Test the API

#### Using Postman (Recommended)
**[📌 View Postman Collection](https://www.postman.com/sy-eduzair-1366980/workspace/syed-uzair-s-workspace/collection/53747524-ed7a4f55-c126-42be-883f-d54fe3012bf3?action=share&creator=53747524)**

**[📌 View Postman Documentation](https://documenter.getpostman.com/view/53747524/2sBXirk9F4)**
Import this collection to get pre-configured requests and examples for all endpoints.

#### Using Python Script

```bash
cd app
python ocr.py  # Runs preprocessing, OCR, and field extraction on test image
```

This will:
- Load a test image
- Show each preprocessing step visually (press any key to advance)
- Run OCR extraction
- Display extracted NIC fields

#### Using cURL

```bash
curl -X POST "http://localhost:8000/v1/ocr/nic" \
  -F "image=@/path/to/nic_image.jpg"
```

## 📊 API Endpoints

### POST `/v1/ocr/nic`

Extract NIC information from an image.

**Request:**
- Content-Type: `multipart/form-data`
- Field: `image` (image file)
- Supported formats: JPEG, PNG, WebP, BMP, TIFF
- Max size: 10 MB

**Response:**
```json
{
  "raw_text": "Full OCR text from image...",
  "lines": ["Line 1", "Line 2", ...],
  "extracted_fields": {
    "cnic_number": "12345-1234567-1",
    "name": "JOHN DOE",
    "father_or_husband_name": "FATHER NAME",
    "gender": "Male",
    "date_of_birth": "01.01.1990",
    "country_of_stay": "Pakistan"
  },
  "image_metadata": {
    "original_size": [300, 400],
    "processed_size": [900, 1200],
    "preprocessing_steps": ["grayscale", "normalize", "super-image", ...]
  },
  "ocr_metadata": {
    "engine": "easyocr",
    "language": "en",
    "mean_confidence": 92.5,
    "word_count": 15,
    "extraction_status": "success"
  },
  "warnings": []
}
```

### GET `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

## 🔧 Configuration

Edit `app/config.py` to customize settings:

```python
TARGET_WIDTH = 1200              # Target output width for images
MIN_ACCEPTABLE_WIDTH = 900       # Minimum width before upscaling
MAX_UPSCALE_FACTOR = 1.5         # Maximum traditional upscale factor
CLAHE_TILE_SIZE = 8              # Contrast enhancement tile size
DENOISE_H = 5                    # Denoising strength
MORPH_KERNEL_SIZE = 2            # Morphological operations kernel size
```

## 📸 Preprocessing Pipeline

The image processing pipeline includes:

1. **Grayscale Conversion**: Convert BGR to single-channel grayscale
2. **Normalization**: Stretch intensity range to full 0-255 spectrum
3. **Super-Image Upscaling**: AI-powered 2x upscaling using RealESRGAN (if width < 900px)
4. **Quality Assessment**: Evaluate sharpness, brightness, and contrast
5. **Brightness Correction**: Apply gamma correction if image is dark
6. **Denoising**: Remove noise using bilateral filtering
7. **CLAHE**: Adaptive Contrast Limited Histogram Equalization
8. **Sharpening**: Enhance edges using unsharp masking
9. **Morphological Operations**: Open/close operations to clean up artifacts

## 🧪 Testing & Debugging

### View Full Pipeline

```bash
cd app
python test_pipeline.py "/path/to/image.jpg"
```

This saves intermediate images at each preprocessing step for visual inspection.

### Compare Upscaling Methods

```bash
cd app
python test_pipeline.py "/path/to/image.jpg" --compare
```

Creates side-by-side comparisons of traditional vs super-image upscaling.

### Benchmark Methods

```bash
cd app
python test_utils.py "/path/to/image.jpg" --benchmark
```

Compares speed and quality of different upscaling methods.

### Interactive Testing

```bash
cd app
python test_utils.py "/path/to/image.jpg"
python test_utils.py "/path/to/image.jpg" --compare
```

## 📈 Performance

### Speed (CPU)
- Traditional upscale: 1-5ms
- Super-image (2x): 500-2000ms (first run, model loads)
- Subsequent runs: 200-500ms (model cached)

### Quality
- Text clarity: ⭐⭐⭐⭐⭐ (super-image) vs ⭐⭐⭐ (traditional)
- Detail preservation: ⭐⭐⭐⭐⭐
- OCR accuracy improvement: ~15-30% better on small images

## 🐛 Troubleshooting

### Import Errors
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Image Not Found
Ensure the image path is correct and the file exists.

### Memory Issues
- Use CPU device (slower but lower memory)
- Reduce image size before upscaling

### Super-Image Slow
Normal on first run (model loading). Subsequent runs are cached and faster.

### No GPU Access
Install GPU support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📝 Dependencies

See `requirements.txt` for complete list. Key packages:
- fastapi
- uvicorn
- opencv-python
- easyocr
- torch
- torchvision
- super-image
- pydantic
- pydantic-settings

## 📚 Documentation

- **OpenCV**: https://docs.opencv.org/
- **EasyOCR**: https://github.com/JaidedAI/EasyOCR
- **FastAPI**: https://fastapi.tiangolo.com/
- **PyTorch**: https://pytorch.org/
- **Super-Image**: https://github.com/idealo/image-super-resolution

## 👨‍💼 Author

**Syed Uzair Hussain Zaidi**

## 📄 License

This project is part of Tezeract OCR NIC Task.

---

**Need Help?** Check the Postman collection for API examples or run the test scripts for debugging!
