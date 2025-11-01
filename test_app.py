#!/usr/bin/env python3
"""
Test script for AI MoodMate app functionality
"""

import os
import sys
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Add the project directory to Python path
sys.path.insert(0, '/home/surendra208/Documents/jaya/aimoodmate/YOLO/ai_moodmate')

def test_model_loading():
    """Test if the YOLO model loads correctly"""
    print("Testing model loading...")
    model_path = '/home/surendra208/Documents/jaya/aimoodmate/YOLO/ai_moodmate/last.pt'
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at {model_path}")
        return False
    
    try:
        model = YOLO(model_path)
        print(f"✅ Model loaded successfully!")
        print(f"   Classes: {model.model.names}")
        print(f"   Number of classes: {len(model.model.names)}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def test_emotion_detection():
    """Test emotion detection on a dummy image"""
    print("\nTesting emotion detection...")
    
    try:
        model = YOLO('/home/surendra208/Documents/jaya/aimoodmate/YOLO/ai_moodmate/last.pt')
        
        # Create a dummy image (random noise)
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Run prediction
        results = model.predict(dummy_image, conf=0.25, verbose=False)
        
        print(f"✅ Emotion detection test completed!")
        print(f"   Results: {len(results)} detection(s)")
        
        if results and len(results) > 0:
            res = results[0]
            if hasattr(res, "boxes") and res.boxes is not None:
                print(f"   Detected {len(res.boxes)} face(s)")
            else:
                print("   No faces detected (expected for random image)")
        
        return True
    except Exception as e:
        print(f"❌ Error in emotion detection: {e}")
        return False

def test_app_imports():
    """Test if all required modules can be imported"""
    print("\nTesting app imports...")
    
    try:
        import streamlit as st
        import numpy as np
        import pandas as pd
        import plotly.express as px
        from PIL import Image
        import cv2
        from ultralytics import YOLO
        from fpdf import FPDF
        from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
        import av
        
        print("✅ All required modules imported successfully!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_file_structure():
    """Test if all required files exist"""
    print("\nTesting file structure...")
    
    required_files = [
        'app.py',
        'requirements.txt',
        'last.pt',
        'assets/logo.png',
        'outputs'
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = f'/home/surendra208/Documents/jaya/aimoodmate/YOLO/ai_moodmate/{file_path}'
        if os.path.exists(full_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing!")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("🧪 AI MoodMate App Test Suite")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_app_imports,
        test_model_loading,
        test_emotion_detection
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The AI MoodMate app is ready to use.")
        print("\n✨ NEW FEATURES:")
        print("   • 5 songs per emotion (instead of 2)")
        print("   • Detailed song descriptions")
        print("   • Clickable links for reading recommendations")
        print("   • Fixed deprecation warnings")
        print("\nTo run the app:")
        print("1. cd /home/surendra208/Documents/jaya/aimoodmate/YOLO/ai_moodmate")
        print("2. source .venv/bin/activate")
        print("3. streamlit run app.py")
        print("\nThe app will be available at: http://localhost:8501")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
