# AI MoodMate - Emotion Detection and Music Recommendation System

A professional Streamlit application that detects facial emotions using a trained YOLOv11 model and provides personalized music recommendations, wellness tips, and therapy resources.

## Features

- **Multi-Input Support**: Image, Video, and Live Webcam inputs
- **Real-time Emotion Detection**: Powered by YOLOv11 with 9 emotion classes
- **Smart Analytics**: Confidence-weighted emotion aggregation and percentage calculations
- **Personalized Recommendations**: 
  - 5 YouTube music links per emotion with detailed explanations
  - Reading and mindfulness suggestions with clickable links
  - Therapy and counseling resources
- **PDF Reports**: Downloadable session summaries
- **Professional UI**: Clean, responsive interface with progress indicators

## Emotion Classes

The model detects 9 different emotions:
- Angry
- Contempt  
- Disgust
- Fear
- Happy
- Natural
- Sad
- Sleepy
- Surprised

## Project Structure

```
ai_moodmate/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── test_app.py          # Test suite for verification
├── assets/
│   └── logo.png         # App logo
├── outputs/             # Generated PDFs and temporary files
└── last.pt             # Trained YOLOv11 model
```

## Installation

1. **Clone/Navigate to the project directory:**
   ```bash
   cd /home/surendra208/Documents/jaya/aimoodmate/YOLO/ai_moodmate
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python test_app.py
   ```

## Usage

1. **Start the application:**
   ```bash
   source .venv/bin/activate
   streamlit run app.py
   ```

2. **Access the app:**
   - Open your browser and go to `http://localhost:8501`
   - Choose input mode: Image, Video, or Live Webcam
   - Adjust confidence threshold if needed
   - Click "Run Detection" to start analysis

3. **View results:**
   - See detected emotions with bounding boxes
   - Review emotion percentage chart
   - Get personalized recommendations
   - Download PDF summary

## Input Modes

### Image Mode
- Upload JPG, JPEG, or PNG images
- Instant emotion detection and analysis
- Perfect for single photo analysis

### Video Mode  
- Upload MP4, MOV, AVI, or MKV videos
- Frame-by-frame emotion analysis
- Progress indicator and preview frames
- Aggregated emotion percentages

### Live Webcam Mode
- Real-time camera feed analysis
- Continuous emotion detection
- Perfect for live mood monitoring
- Requires camera permissions

## Recommendations System

### Music Recommendations
- Curated YouTube links for each emotion
- Short explanations for why each song fits
- Mix of international and Indian music
- Links open in new tabs

### Reading & Mindfulness
- Books and articles tailored to emotions
- Quick mindfulness exercises
- Evidence-based suggestions
- Practical implementation tips

### Therapy Resources
- India-specific mental health resources
- Professional counseling services
- Clinical support networks
- Non-clinical wellness pointers

## PDF Reports

Each session generates a comprehensive PDF including:
- Session timestamp and input mode
- Detailed emotion percentages
- Dominant emotion identification
- All recommendations with links
- Therapy resource information

## Technical Details

- **Model**: YOLOv11 (Ultralytics)
- **Framework**: Streamlit 1.50.0
- **Computer Vision**: OpenCV 4.12.0
- **Charts**: Plotly 6.3.1
- **PDF Generation**: FPDF 1.7.2
- **Webcam**: streamlit-webrtc 0.63.11

## Model Information

- **Classes**: 9 emotion categories
- **Format**: PyTorch (.pt)
- **Architecture**: YOLOv11
- **Training**: Custom emotion detection dataset

## Troubleshooting

### Common Issues

1. **Model not loading:**
   - Ensure `last.pt` exists in project root
   - Check file permissions
   - Verify Ultralytics installation

2. **Webcam not working:**
   - Grant camera permissions in browser
   - Check if camera is being used by another app
   - Try refreshing the page

3. **Dependencies issues:**
   - Use the exact versions in requirements.txt
   - Create fresh virtual environment
   - Run `python test_app.py` to verify

### Performance Tips

- Lower confidence threshold for more detections
- Use smaller video files for faster processing
- Close other applications to free up resources
- Ensure good lighting for webcam mode

## Development

### Running Tests
```bash
python test_app.py
```

### Adding New Emotions
1. Retrain the YOLOv11 model with new classes
2. Update `EMOTION_CLASSES` in app.py
3. Add recommendations for new emotions
4. Update test suite

### Customizing Recommendations
Edit the recommendation dictionaries in app.py:
- `YOUTUBE_SONGS`: Music recommendations
- `READING_MINDFULNESS`: Books and exercises
- `THERAPY_RESOURCES`: Support services

## License

This project is for educational and research purposes. Please ensure you have appropriate permissions for any models or datasets used.

## Support

For technical issues or questions about the AI MoodMate system, please refer to the troubleshooting section or check the test suite output for specific error messages.

---

**Note**: This application provides supportive recommendations and is not a substitute for professional medical or psychological advice. Always consult qualified professionals for mental health concerns.
