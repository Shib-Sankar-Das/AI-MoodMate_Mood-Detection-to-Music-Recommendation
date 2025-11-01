import os
import io
import time
from datetime import datetime
from typing import Dict, List, Tuple

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import cv2

from ultralytics import YOLO

from fpdf import FPDF

# Webcam
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av

# Must be the first Streamlit command
st.set_page_config(page_title="AI MoodMate", page_icon="🧠", layout="wide")

# Custom CSS for professional styling
def load_custom_css():
    st.markdown("""
    <style>
    /* Main App Styling */
    .main {
        padding: 1rem 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    /* Header Styling */
    .header-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .app-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .app-subtitle {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 1rem;
    }
    
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Card Styling */
    .card {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        border-left: 5px solid #667eea;
    }
    
    /* Emotion-specific colors */
    .emotion-angry { border-left-color: #e74c3c; }
    .emotion-fear { border-left-color: #9b59b6; }
    .emotion-sad { border-left-color: #3498db; }
    .emotion-happy { border-left-color: #f39c12; }
    .emotion-surprised { border-left-color: #e67e22; }
    .emotion-disgust { border-left-color: #27ae60; }
    .emotion-contempt { border-left-color: #95a5a6; }
    .emotion-natural { border-left-color: #34495e; }
    .emotion-sleepy { border-left-color: #8e44ad; }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Progress Bar Styling */
    .progress-container {
        background: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Chart Container */
    .chart-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    /* Recommendations Styling */
    .recommendation-section {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    /* Footer Styling */
    .footer {
        text-align: center;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        margin-top: 2rem;
        color: #333;
        font-size: 0.9rem;
    }
    
    .footer p {
        margin: 0.5rem 0;
        line-height: 1.6;
    }
    
    .footer strong {
        color: #667eea;
        font-size: 1.1rem;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main {
            padding: 0.5rem;
        }
        .app-title {
            font-size: 2rem;
        }
        .header-container {
            padding: 1rem;
        }
    }
    
    /* Loading Animation */
    .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255,255,255,.3);
        border-radius: 50%;
        border-top-color: #667eea;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Success/Error Messages */
    .success-message {
        background: linear-gradient(45deg, #27ae60, #2ecc71);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .warning-message {
        background: linear-gradient(45deg, #f39c12, #e67e22);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ----------------------------
# CONFIG
# ----------------------------
APP_TITLE = "AI MoodMate"

# Load custom CSS
load_custom_css()

# ----------------------------
# SESSION MANAGEMENT & MOOD HISTORY
# ----------------------------
def initialize_session_data():
    """Initialize session data for mood tracking"""
    if 'mood_history' not in st.session_state:
        st.session_state.mood_history = []
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = {
            'favorite_genres': [],
            'preferred_activities': [],
            'wellness_goals': [],
            'session_count': 0
        }
    if 'current_session' not in st.session_state:
        st.session_state.current_session = {
            'start_time': None,
            'emotions_detected': [],
            'recommendations_given': [],
            'breathing_exercises_completed': 0
        }

def save_mood_session(emotion_data, recommendations, input_mode):
    """Save current mood session to history"""
    session_data = {
        'timestamp': datetime.now(),
        'date': datetime.now().strftime('%Y-%m-%d'),
        'time': datetime.now().strftime('%H:%M'),
        'input_mode': input_mode,
        'dominant_emotion': emotion_data.get('dominant', 'unknown'),
        'emotion_percentages': emotion_data.get('percentages', {}),
        'recommendations': recommendations,
        'session_id': f"session_{len(st.session_state.mood_history) + 1}"
    }
    
    st.session_state.mood_history.append(session_data)
    st.session_state.user_preferences['session_count'] += 1
    
    # Update user preferences based on interactions
    update_user_preferences(session_data)

def update_user_preferences(session_data):
    """Update user preferences based on session data"""
    # Track favorite emotions (for personalized recommendations)
    dominant = session_data['dominant_emotion']
    if dominant not in st.session_state.user_preferences['favorite_genres']:
        st.session_state.user_preferences['favorite_genres'].append(dominant)
    
    # Keep only last 10 preferences to avoid clutter
    if len(st.session_state.user_preferences['favorite_genres']) > 10:
        st.session_state.user_preferences['favorite_genres'] = st.session_state.user_preferences['favorite_genres'][-10:]

def get_mood_insights():
    """Generate insights from mood history"""
    if not st.session_state.mood_history:
        return None
    
    # Calculate mood trends
    recent_sessions = st.session_state.mood_history[-7:]  # Last 7 sessions
    emotion_counts = {}
    
    for session in recent_sessions:
        emotion = session['dominant_emotion']
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    # Find most common emotion
    most_common = max(emotion_counts.items(), key=lambda x: x[1]) if emotion_counts else ('unknown', 0)
    
    # Calculate mood stability (lower variance = more stable)
    emotions = [session['dominant_emotion'] for session in recent_sessions]
    stability_score = len(set(emotions)) / len(emotions) if emotions else 0
    
    return {
        'total_sessions': len(st.session_state.mood_history),
        'most_common_emotion': most_common[0],
        'emotion_frequency': most_common[1],
        'mood_stability': stability_score,
        'recent_trend': emotion_counts,
        'last_session': st.session_state.mood_history[-1] if st.session_state.mood_history else None
    }

# ----------------------------
# PATHS CONFIGURATION
# ----------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, "last.pt")
ASSETS_DIR = os.path.join(PROJECT_DIR, "assets")
OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

EMOTION_CLASSES = ["angry","contempt","disgust","fear","happy","natural","sad","sleepy","surprised"]

# Create a placeholder logo if not present
logo_path = os.path.join(ASSETS_DIR, "logo.png")
if not os.path.exists(logo_path):
    img = Image.new("RGBA", (512, 512), (240, 248, 255, 255))
    img.save(logo_path)

# ----------------------------
# RECOMMENDATION CATALOGS
# (short rationale per item; keep concise)
# ----------------------------

YOUTUBE_SONGS = {
    "happy": [
        ("Pharrell Williams - Happy", "Upbeat rhythm reinforces positive affect and boosts mood naturally", "https://www.youtube.com/watch?v=ZbZSe6N_BXs"),
        ("Katrina Kaif - Nachde Ne Saare", "Energetic tempo matches joyful state and encourages movement", "https://www.youtube.com/watch?v=3PgmGq3oPoE"),
        ("Bruno Mars - Uptown Funk", "High-energy funk that amplifies happiness and confidence", "https://www.youtube.com/watch?v=OPf0YbXqDm0"),
        ("Taylor Swift - Shake It Off", "Empowering lyrics help shake off negativity and embrace joy", "https://www.youtube.com/watch?v=nfWlot6h_JM"),
        ("Arijit Singh - Tum Hi Ho", "Melodic celebration of love and happiness in relationships", "https://www.youtube.com/watch?v=7wtfhZwyrcc"),
    ],
    "sad": [
        ("Coldplay - Fix You", "Gentle build helps emotional release and provides comfort", "https://www.youtube.com/watch?v=k4V3Mo61fJM"),
        ("Arijit Singh - Channa Mereya", "Cathartic lyrics align with sadness and offer emotional validation", "https://www.youtube.com/watch?v=284Ov7ysmfA"),
        ("Adele - Someone Like You", "Powerful ballad that helps process feelings of loss and longing", "https://www.youtube.com/watch?v=hLQl3WQQoQ0"),
        ("Ed Sheeran - Photograph", "Tender melody that acknowledges sadness while offering hope", "https://www.youtube.com/watch?v=nSDgHBxUbVQ"),
        ("Lata Mangeshkar - Lag Ja Gale", "Classic Hindi song that provides emotional catharsis and healing", "https://www.youtube.com/watch?v=0-Ed3qJzqgc"),
    ],
    "angry": [
        ("Linkin Park - Numb", "High energy channels tension safely and validates frustration", "https://www.youtube.com/watch?v=kXYiU_JCYtU"),
        ("Imagine Dragons - Believer", "Percussive drive aids affect regulation and builds resilience", "https://www.youtube.com/watch?v=7wtfhZwyrcc"),
        ("Eminem - Lose Yourself", "Intense rap that channels anger into motivation and determination", "https://www.youtube.com/watch?v=_Yhyp-_hX2s"),
        ("Rage Against The Machine - Killing in the Name", "Aggressive rock that provides safe outlet for anger expression", "https://www.youtube.com/watch?v=bWXazVhlyxQ"),
        ("Badshah - Proper Patola", "High-energy Punjabi track that transforms anger into dance energy", "https://www.youtube.com/watch?v=6ZgKuZqXwbg"),
    ],
    "fear": [
        ("AURORA - Runaway", "Airy vocals reduce perceived threat and create calming atmosphere", "https://www.youtube.com/watch?v=d_HlPboLRL8"),
        ("Prateek Kuhad - Cold/Mess", "Soothing tone lowers arousal and provides emotional safety", "https://www.youtube.com/watch?v=On86kqM1bX4"),
        ("Billie Eilish - Everything I Wanted", "Gentle electronic sounds that help process anxiety and fear", "https://www.youtube.com/watch?v=EgBJmlPo8Xw"),
        ("Bon Iver - Skinny Love", "Minimalist arrangement that creates sense of security and peace", "https://www.youtube.com/watch?v=8j741TUIET0"),
        ("A R Rahman - Mumbai Theme", "Ambient instrumental that provides grounding and reduces anxiety", "https://www.youtube.com/watch?v=DkO5G6j1GIs"),
    ],
    "disgust": [
        ("Daft Punk - Get Lucky", "Clean groove resets affective state and brings positive energy", "https://www.youtube.com/watch?v=5NV6Rdv1a3I"),
        ("Shankar–Ehsaan–Loy - Dil Chahta Hai", "Light, refreshing vibe reorients mood and clears negative feelings", "https://www.youtube.com/watch?v=0-Ed3qJzqgc"),
        ("Kygo - Firestone", "Upbeat tropical house that washes away disgust with positivity", "https://www.youtube.com/watch?v=9Sc-ir2UwGU"),
        ("Calvin Harris - Summer", "Energetic electronic music that transforms negative emotions", "https://www.youtube.com/watch?v=ebXbLfLACGM"),
        ("Arijit Singh - Kesariya", "Melodic Bollywood track that replaces disgust with romantic feelings", "https://www.youtube.com/watch?v=OPf0YbXqDm0"),
    ],
    "surprised": [
        ("OK Go - Here It Goes Again", "Playful novelty complements surprise and encourages exploration", "https://www.youtube.com/watch?v=dTAAsCNK7RA"),
        ("Badshah - Proper Patola", "Festive bounce sustains positive surprise and amplifies excitement", "https://www.youtube.com/watch?v=6ZgKuZqXwbg"),
        ("Panic! At The Disco - High Hopes", "Uplifting anthem that channels surprise into motivation and optimism", "https://www.youtube.com/watch?v=IPXIgEAGe4U"),
        ("The Weeknd - Blinding Lights", "Retro synth-pop that celebrates unexpected positive moments", "https://www.youtube.com/watch?v=4NRXx6U8ABQ"),
        ("Dua Lipa - Levitating", "Futuristic pop that elevates surprise into pure joy and energy", "https://www.youtube.com/watch?v=TUVcZfQe-Kw"),
    ],
    "contempt": [
        ("Daft Punk - Get Lucky", "Clean groove resets affective state and brings positive energy", "https://www.youtube.com/watch?v=5NV6Rdv1a3I"),
        ("Shankar–Ehsaan–Loy - Dil Chahta Hai", "Light, refreshing vibe reorients mood and clears negative feelings", "https://www.youtube.com/watch?v=0-Ed3qJzqgc"),
        ("Kygo - Firestone", "Upbeat tropical house that washes away contempt with positivity", "https://www.youtube.com/watch?v=9Sc-ir2UwGU"),
        ("Calvin Harris - Summer", "Energetic electronic music that transforms negative emotions", "https://www.youtube.com/watch?v=ebXbLfLACGM"),
        ("Arijit Singh - Kesariya", "Melodic Bollywood track that replaces contempt with romantic feelings", "https://www.youtube.com/watch?v=OPf0YbXqDm0"),
    ],
    "natural": [
        ("Ludovico Einaudi - Nuvole Bianche", "Calm piano supports reflection and maintains emotional balance", "https://www.youtube.com/watch?v=kcihcYEOeic"),
        ("A R Rahman - Mumbai Theme", "Ambient flow maintains balance and provides peaceful atmosphere", "https://www.youtube.com/watch?v=DkO5G6j1GIs"),
        ("Max Richter - On The Nature of Daylight", "Gentle orchestral piece that supports neutral emotional state", "https://www.youtube.com/watch?v=rVN1B-tUpgs"),
        ("Ólafur Arnalds - Near Light", "Minimalist composition that enhances focus and inner calm", "https://www.youtube.com/watch?v=4NRXx6U8ABQ"),
        ("Nils Frahm - Says", "Ambient electronic that promotes mindfulness and presence", "https://www.youtube.com/watch?v=TUVcZfQe-Kw"),
    ],
    "sleepy": [
        ("Ludovico Einaudi - Nuvole Bianche", "Calm piano supports relaxation and prepares mind for rest", "https://www.youtube.com/watch?v=kcihcYEOeic"),
        ("Max Richter - On The Nature of Daylight", "Gentle orchestral piece for rest and peaceful sleep preparation", "https://www.youtube.com/watch?v=rVN1B-tUpgs"),
        ("Ólafur Arnalds - Near Light", "Soft melodies that gently guide toward sleep and relaxation", "https://www.youtube.com/watch?v=4NRXx6U8ABQ"),
        ("Nils Frahm - Says", "Ambient sounds that create perfect sleep-inducing atmosphere", "https://www.youtube.com/watch?v=TUVcZfQe-Kw"),
        ("Brian Eno - An Ending (Ascent)", "Ethereal ambient music that promotes deep relaxation and sleep", "https://www.youtube.com/watch?v=OPf0YbXqDm0"),
    ],
}

READING_MINDFULNESS = {
    "happy": [
        ("The Science of Happiness - Greater Good", "Free research-based happiness practices", "https://greatergood.berkeley.edu/topic/happiness"),
        ("Gratitude Journaling Guide - Mindful.org", "Free step-by-step gratitude practice", "https://www.mindful.org/how-to-start-a-gratitude-practice/"),
        ("Positive Psychology Exercises - Verywell Mind", "Free activities to boost wellbeing", "https://www.verywellmind.com/positive-psychology-exercises-2795045"),
        ("Savoring Positive Moments - Psychology Today", "Free techniques to extend joy", "https://www.psychologytoday.com/us/blog/fulfillment-any-age/201201/how-savor-positive-moments"),
        ("Flow State Activities - Mindful.org", "Free guide to finding your flow", "https://www.mindful.org/how-to-find-your-flow-state/"),
    ],
    "sad": [
        ("Coping with Sadness - Mayo Clinic", "Free evidence-based strategies for low mood", "https://www.mayoclinic.org/diseases-conditions/depression/symptoms-causes/syc-20356007"),
        ("Self-Compassion During Difficult Times - Mindful.org", "Free practices for self-kindness", "https://www.mindful.org/loving-kindness-meditation/"),
        ("Grief and Loss Resources - Psychology Today", "Free support for processing sadness", "https://www.psychologytoday.com/us/basics/grief"),
        ("Depression Support Techniques - Verywell Mind", "Free professional strategies", "https://www.healthline.com/health/breathing-exercise"),
        ("Mindful Depression Management - Mindful.org", "Free mindfulness-based approaches", "https://www.mindful.org/mindful-depression-management/"),
    ],
    "angry": [
        ("Anger Management Techniques - Mayo Clinic", "Evidence-based strategies for managing anger", "https://www.mayoclinic.org/healthy-lifestyle/adult-health/in-depth/anger-management/art-20045434"),
        ("Mindful Anger: A Guide to Emotional Regulation", "Free article on mindfulness-based anger management", "https://www.mindful.org/mindful-anger/"),
        ("Psychology Today: Understanding Anger", "Professional insights on anger psychology", "https://www.psychologytoday.com/us/basics/anger"),
        ("Breathing Exercises for Anger - Healthline", "Immediate techniques to calm anger", "https://www.healthline.com/health/box-breathing"),
        ("The Science of Anger - Verywell Mind", "Understanding the biology and psychology of anger", "https://www.verywellmind.com/what-is-anger-5120208"),
    ],
    "fear": [
        ("Managing Anxiety and Fear - Mayo Clinic", "Free evidence-based anxiety management", "https://www.mayoclinic.org/diseases-conditions/anxiety/symptoms-causes/syc-20350961"),
        ("Breathing Exercises for Anxiety - Healthline", "Free immediate relief techniques", "https://www.healthline.com/health/grounding-techniques"),
        ("Progressive Muscle Relaxation - Verywell Mind", "Free step-by-step relaxation guide", "https://www.verywellmind.com/progressive-muscle-relaxation-2584454"),
        ("Exposure Therapy Techniques - Psychology Today", "Free guide to facing fears gradually", "https://www.mindful.org/mindfulness-based-stress-reduction/"),
        ("Calm Breathing Techniques - Mindful.org", "Free breathing practices for anxiety", "https://www.mindful.org/how-to-practice-mindful-breathing/"),
    ],
    "disgust": [
        ("Understanding Disgust - Psychology Today", "Free article on the psychology of disgust", "https://www.psychologytoday.com/us/basics/disgust"),
        ("Body Image and Self-Compassion - Mindful.org", "Free resources for body acceptance", "https://www.mindful.org/how-to-practice-mindful-walking/"),
        ("Grounding Techniques for Trauma - Verywell Mind", "Free techniques for emotional regulation", "https://www.verywellmind.com/grounding-techniques-for-trauma-5206278"),
        ("Mindful Eating Guide - Mindful.org", "Free practices for healthy food relationship", "https://www.mindful.org/compassion-meditation/"),
        ("Self-Compassion Exercises - Self-Compassion.org", "Free practices by Dr. Kristin Neff", "https://self-compassion.org/category/exercises/"),
    ],
    "surprised": [
        ("Adaptability and Resilience - Psychology Today", "Free guide to handling unexpected changes", "https://www.psychologytoday.com/us/basics/resilience"),
        ("Mindful Curiosity Practices - Mindful.org", "Free techniques to channel surprise positively", "https://www.psychologytoday.com/us/blog/the-happiness-project/201503/3-new-things"),
        ("Change Management Strategies - Verywell Mind", "Free approaches to navigate transitions", "https://www.verywellmind.com/how-to-deal-with-change-3145028"),
        ("Openness to Experience - Greater Good", "Free practices for flexibility and growth", "https://www.mindful.org/the-power-of-curiosity/"),
        ("Embracing Uncertainty - Mindful.org", "Free mindfulness practices for life's surprises", "https://www.mindful.org/how-to-embrace-uncertainty/"),
    ],
    "contempt": [
        ("Overcoming Judgment and Contempt - Mindful.org", "Free guide to reducing judgmental thinking", "https://www.mindful.org/overcoming-judgment/"),
        ("The Psychology of Contempt - Psychology Today", "Understanding contempt and its effects", "https://www.mindful.org/how-to-practice-mindful-walking/"),
        ("Compassion Meditation Guide - Greater Good", "Free practices to develop compassion", "https://greatergood.berkeley.edu/topic/compassion"),
        ("Mindful Walking Practice - Mindful.org", "Step-by-step guide to mindful movement", "https://www.mindful.org/compassion-meditation/"),
        ("Radical Acceptance - Psychology Today", "Free article on accepting reality without judgment", "https://www.psychologytoday.com/us/blog/compassion-matters/201307/radical-acceptance"),
    ],
    "natural": [
        ("Mindfulness for Beginners - Mindful.org", "Free comprehensive mindfulness guide", "https://www.mindful.org/how-to-practice-mindfulness/"),
        ("Body Scan Meditation - Greater Good", "Free guided body awareness practice", "https://www.mindful.org/how-to-practice-mindful-breathing/"),
        ("Loving-Kindness Meditation - Mindful.org", "Free compassion cultivation practice", "https://www.mindful.org/how-to-practice-loving-kindness-meditation/"),
        ("Daily Mindfulness Routine - Verywell Mind", "Free simple practices for balance", "https://www.mindful.org/how-to-meditate/"),
        ("Mindfulness Research - Greater Good", "Free scientific articles on mindfulness benefits", "https://greatergood.berkeley.edu/topic/mindfulness"),
    ],
    "sleepy": [
        ("Sleep Hygiene Guide - Mayo Clinic", "Free evidence-based sleep improvement", "https://www.mayoclinic.org/healthy-lifestyle/adult-health/in-depth/sleep/art-20048379"),
        ("Relaxation Techniques for Sleep - Healthline", "Free practices for better rest", "https://www.mayoclinic.org/healthy-lifestyle/stress-management/in-depth/progressive-muscle-relaxation/art-20045354"),
        ("Circadian Rhythm Optimization - Verywell Mind", "Free guide to natural sleep cycles", "https://www.sleepfoundation.org/sleep-hygiene"),
        ("Insomnia Management - Psychology Today", "Free strategies for sleep difficulties", "https://www.headspace.com/sleep"),
        ("Mindful Sleep Practices - Mindful.org", "Free meditation for better sleep", "https://www.mindful.org/how-to-practice-mindful-sleep/"),
    ],
}

THERAPY_RESOURCES = [
    ("NIMHANS (India)", "National mental health institute with clinical services", "https://www.nimhans.ac.in/"),
    ("iCALL Tata Institute of Social Sciences", "Professional counseling helpline", "https://icallhelpline.org/"),
    ("Fortis Mental Health", "Counseling & psychiatry network", "https://www.fortishealthcare.com/india/clinical-speciality/mental-health-and-behavioural-sciences"),
]

# Breathing exercises for each emotion with scientific references
BREATHING_EXERCISES = {
    "angry": {
        "name": "4-7-8 Calming Breath",
        "description": "Slow breathing technique to activate parasympathetic nervous system and reduce anger",
        "technique": "Inhale for 4 counts, hold for 7 counts, exhale for 8 counts",
        "duration": "5-10 minutes",
        "reference": "Weil, A. (2012). Breathing: The Master Key to Self Healing. Evidence shows slow breathing reduces cortisol and activates relaxation response.",
        "reference_link": "https://www.drweil.com/health-wellness/body-mind-spirit/stress-anxiety/breathing-exercises-4-7-8-breath/",
        "steps": [
            "Sit comfortably with spine straight",
            "Place tip of tongue behind upper front teeth",
            "Exhale completely through mouth",
            "Close mouth, inhale through nose for 4 counts",
            "Hold breath for 7 counts", 
            "Exhale through mouth for 8 counts",
            "Repeat 4-8 cycles"
        ]
    },
    "contempt": {
        "name": "Heart-Centered Breathing",
        "description": "Compassion-focused breathing to dissolve judgmental feelings and cultivate empathy",
        "technique": "Breathe into heart center while focusing on compassion",
        "duration": "8-12 minutes",
        "reference": "Fredrickson, B. (2013). Love 2.0: How Our Supreme Emotion Affects Everything We Feel, Think, Do, and Become. Heart-focused breathing increases positive emotions.",
        "reference_link": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3156028/",
        "steps": [
            "Place hand over heart center",
            "Breathe slowly and deeply",
            "With each inhale, imagine breathing in compassion",
            "With each exhale, release judgment and criticism",
            "Visualize warm, loving energy in your heart",
            "Extend compassion to yourself and others",
            "Continue for 8-12 minutes"
        ]
    },
    "disgust": {
        "name": "Cleansing Breath",
        "description": "Purifying breathing technique to release negative emotions and refresh mental state",
        "technique": "Deep inhales followed by forceful exhales to cleanse",
        "duration": "6-8 minutes",
        "reference": "Brown, R. P., & Gerbarg, P. L. (2005). Sudarshan Kriya yogic breathing in the treatment of stress, anxiety, and depression. Journal of Alternative and Complementary Medicine.",
        "reference_link": "https://pubmed.ncbi.nlm.nih.gov/16332104/",
        "steps": [
            "Sit upright with shoulders relaxed",
            "Take deep breath through nose",
            "Hold for 2-3 seconds",
            "Exhale forcefully through mouth",
            "Imagine releasing all negativity",
            "Repeat 8-12 times",
            "End with 3 gentle breaths"
        ]
    },
    "fear": {
        "name": "Grounding Breath",
        "description": "Stabilizing breathing to calm nervous system and reduce anxiety",
        "technique": "Slow, deep breathing with grounding visualization",
        "duration": "10-15 minutes",
        "reference": "Jerath, R., et al. (2006). Physiology of long pranayamic breathing: Neural respiratory elements may provide a mechanism. Journal of Applied Physiology.",
        "reference_link": "https://pubmed.ncbi.nlm.nih.gov/16282441/",
        "steps": [
            "Sit with feet flat on ground",
            "Place hands on thighs",
            "Breathe slowly through nose",
            "Feel connection to earth",
            "With each breath, imagine roots growing down",
            "Focus on present moment",
            "Continue until feeling grounded"
        ]
    },
    "happy": {
        "name": "Joy Amplification Breath",
        "description": "Energizing breathing to enhance positive emotions and boost mood",
        "technique": "Rhythmic breathing with joyful visualization",
        "duration": "5-8 minutes",
        "reference": "Kok, B. E., et al. (2013). How positive emotions build physical health. Psychological Science.",
        "reference_link": "https://pubmed.ncbi.nlm.nih.gov/23527591/",
        "steps": [
            "Stand or sit with open posture",
            "Smile gently",
            "Breathe in joy and gratitude",
            "Exhale spreading happiness",
            "Visualize golden light filling your body",
            "Feel energy expanding outward",
            "Share joy with the world"
        ]
    },
    "natural": {
        "name": "Mindful Breathing",
        "description": "Present-moment awareness breathing for emotional balance and clarity",
        "technique": "Natural breathing with mindful attention",
        "duration": "10-20 minutes",
        "reference": "Kabat-Zinn, J. (1990). Full Catastrophe Living: Using the Wisdom of Your Body and Mind to Face Stress, Pain, and Illness.",
        "reference_link": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3679190/",
        "steps": [
            "Sit comfortably",
            "Close eyes gently",
            "Notice natural breath",
            "Don't change breathing",
            "Observe sensations",
            "When mind wanders, return to breath",
            "Practice non-judgmental awareness"
        ]
    },
    "sad": {
        "name": "Comforting Breath",
        "description": "Gentle, nurturing breathing to provide emotional support and healing",
        "technique": "Soft, slow breathing with self-compassion",
        "duration": "12-15 minutes",
        "reference": "Neff, K. (2011). Self-Compassion: The Proven Power of Being Kind to Yourself. Compassionate breathing reduces depression symptoms.",
        "reference_link": "https://self-compassion.org/",
        "steps": [
            "Lie down or sit comfortably",
            "Place hand on heart",
            "Breathe softly and slowly",
            "With each breath, offer yourself kindness",
            "Imagine warm, healing light",
            "Allow emotions to flow naturally",
            "Practice self-acceptance"
        ]
    },
    "sleepy": {
        "name": "Energizing Breath",
        "description": "Invigorating breathing to increase alertness and mental clarity",
        "technique": "Quick, energizing breaths followed by deep inhales",
        "duration": "3-5 minutes",
        "reference": "Brown, R. P., & Gerbarg, P. L. (2009). Yoga breathing, meditation, and longevity. Annals of the New York Academy of Sciences.",
        "reference_link": "https://pubmed.ncbi.nlm.nih.gov/19673776/",
        "steps": [
            "Sit upright",
            "Take 3 quick breaths through nose",
            "Hold breath briefly",
            "Exhale slowly",
            "Repeat 5-7 times",
            "End with 3 deep breaths",
            "Feel energy and alertness"
        ]
    },
    "surprised": {
        "name": "Centering Breath",
        "description": "Balancing breathing to process unexpected emotions and regain equilibrium",
        "technique": "Equal-length inhales and exhales for balance",
        "duration": "6-10 minutes",
        "reference": "Gerbarg, P. L., & Brown, R. P. (2005). Yoga and neuro-psychiatric disorders. International Journal of Yoga.",
        "reference_link": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3193654/",
        "steps": [
            "Sit in comfortable position",
            "Breathe in for 4 counts",
            "Hold for 4 counts",
            "Exhale for 4 counts",
            "Hold empty for 4 counts",
            "Repeat equal rhythm",
            "Feel centered and balanced"
        ]
    }
}

# ----------------------------
# Utility functions
# ----------------------------

@st.cache_resource(show_spinner=False)
def load_model():
    return YOLO(MODEL_PATH)

def bgr_to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def rgb_to_bgr(img_rgb):
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

def draw_detections(image_rgb, results, conf_threshold=0.25):
    """Draw boxes with labels on the image."""
    img = image_rgb.copy()
    if results and len(results) > 0:
        res = results[0]
        if hasattr(res, "boxes") and res.boxes is not None:
            for box in res.boxes:
                conf = float(box.conf[0].item()) if box.conf is not None else 0.0
                if conf < conf_threshold:
                    continue
                cls_id = int(box.cls[0].item()) if box.cls is not None else -1
                label = EMOTION_CLASSES[cls_id] if 0 <= cls_id < len(EMOTION_CLASSES) else "unknown"
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                text = f"{label} {conf:.2f}"
                cv2.putText(img, text, (x1, max(0, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(img, text, (x1, max(0, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    return img

def accumulate_emotions(results, weights: Dict[str, float]):
    """Update weights dict with confidence-weighted counts."""
    if not results or len(results) == 0:
        return
    res = results[0]
    if hasattr(res, "boxes") and res.boxes is not None:
        for box in res.boxes:
            conf = float(box.conf[0].item()) if box.conf is not None else 0.0
            cls_id = int(box.cls[0].item()) if box.cls is not None else -1
            if 0 <= cls_id < len(EMOTION_CLASSES):
                label = EMOTION_CLASSES[cls_id]
                weights[label] = weights.get(label, 0.0) + conf

def normalize_percentages(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values()) if weights else 0.0
    if total <= 0:
        return {k: 0.0 for k in EMOTION_CLASSES}
    return {k: round((weights.get(k, 0.0) / total) * 100.0, 2) for k in EMOTION_CLASSES}

def dominant_emotion(percentages: Dict[str, float]) -> str:
    if not percentages:
        return "natural"
    return max(percentages.items(), key=lambda x: x[1])[0]

def recommend_content(emotion: str):
    songs = YOUTUBE_SONGS.get(emotion, YOUTUBE_SONGS["natural"])
    reads = READING_MINDFULNESS.get(emotion, READING_MINDFULNESS["natural"])
    breathing = BREATHING_EXERCISES.get(emotion, BREATHING_EXERCISES["natural"])
    return songs, reads, THERAPY_RESOURCES, breathing

def get_personalized_recommendations(emotion: str):
    """Get personalized recommendations based on user history and preferences"""
    base_songs, base_reads, therapy, breathing = recommend_content(emotion)
    
    # Get user preferences
    user_prefs = st.session_state.user_preferences
    insights = get_mood_insights()
    
    # Personalize recommendations based on history
    personalized_songs = personalize_songs(base_songs, emotion, user_prefs, insights)
    personalized_reads = personalize_reads(base_reads, emotion, user_prefs, insights)
    
    return personalized_songs, personalized_reads, therapy, breathing

def personalize_songs(base_songs, emotion, user_prefs, insights):
    """Personalize song recommendations based on user history"""
    personalized = []
    
    # Add personalized context based on user's mood patterns
    if insights and insights['total_sessions'] > 3:
        if insights['mood_stability'] < 0.5:  # Unstable mood patterns
            personalized.append((
                "🎯 Personalized Pick: Calming Focus Music",
                f"Based on your mood patterns, you might benefit from calming music to help stabilize your emotions.",
                "https://www.youtube.com/results?search_query=calming+focus+music"
            ))
        
        if insights['most_common_emotion'] == emotion:
            personalized.append((
                "📈 Trend-Based Recommendation",
                f"This emotion appears frequently in your sessions. Here are some specialized tracks for deeper exploration.",
                "https://www.youtube.com/results?search_query=emotional+healing+music"
            ))
    
    # Add session count milestone recommendations
    session_count = user_prefs['session_count']
    if session_count > 0 and session_count % 5 == 0:  # Every 5 sessions
        personalized.append((
            f"🏆 Milestone Achievement: {session_count} Sessions!",
            f"Congratulations on your {session_count}th session! Here's a special recommendation for your dedication to emotional wellness.",
            "https://www.youtube.com/results?search_query=motivational+wellness+music"
        ))
    
    # Combine base recommendations with personalized ones
    return personalized + base_songs[:3]  # Limit to 3 base + personalized

def personalize_reads(base_reads, emotion, user_prefs, insights):
    """Personalize reading recommendations based on user history"""
    personalized = []
    
    # Add personalized context based on user's journey
    if insights and insights['total_sessions'] > 2:
        if insights['mood_stability'] < 0.4:  # Very unstable patterns
            personalized.append((
                "🧠 Advanced Emotional Regulation Guide",
                "Based on your emotional patterns, this advanced guide can help you develop stronger emotional regulation skills.",
                "https://www.psychologytoday.com/us/basics/emotional-regulation"
            ))
        
        if insights['total_sessions'] >= 10:
            personalized.append((
                "📚 Deep Dive: Emotional Intelligence Mastery",
                "With your consistent practice, you're ready for advanced emotional intelligence techniques.",
                "https://www.verywellmind.com/what-is-emotional-intelligence-2795423"
            ))
    
    # Add progress-based recommendations
    session_count = user_prefs['session_count']
    if session_count >= 7:
        personalized.append((
            "🌟 Weekly Wellness Reflection",
            "You've been practicing for a week! Here's a guide to reflect on your emotional growth.",
            "https://www.mindful.org/how-to-practice-mindfulness/"
        ))
    
    return personalized + base_reads[:3]  # Limit to 3 base + personalized

def display_breathing_exercise(emotion: str):
    """Display simple breathing exercise for the detected emotion"""
    st.markdown("### 🫁 Breathing Exercise")
    
    # Simple breathing instructions based on emotion
    if emotion == "angry":
        st.info("**4-7-8 Calming Breath** - Slow breathing to reduce anger")
        st.markdown("**Scientific Reference:** [Dr. Andrew Weil's 4-7-8 Technique](https://www.drweil.com/health-wellness/body-mind-spirit/stress-anxiety/breathing-exercises-4-7-8-breath/)")
        
        st.markdown("**Practice Steps:**")
        st.markdown("1. ✅ Sit comfortably with spine straight")
        st.markdown("2. ✅ Place tip of tongue behind upper front teeth") 
        st.markdown("3. ✅ Exhale completely through mouth")
        st.markdown("4. ✅ Close mouth, inhale through nose for 4 counts")
        st.markdown("5. ✅ Hold breath for 7 counts")
        st.markdown("6. ✅ Exhale through mouth for 8 counts")
        st.markdown("7. ✅ Repeat 4-8 cycles")
        
    elif emotion == "fear":
        st.info("**Grounding Breath** - Stabilizing breathing to reduce anxiety")
        st.markdown("**Scientific Reference:** [Jerath et al. (2006) - Pranayamic Breathing](https://pubmed.ncbi.nlm.nih.gov/16282441/)")
        
        st.markdown("**Practice Steps:**")
        st.markdown("1. ✅ Sit with feet flat on ground")
        st.markdown("2. ✅ Place hands on thighs")
        st.markdown("3. ✅ Breathe slowly through nose")
        st.markdown("4. ✅ Feel connection to earth")
        st.markdown("5. ✅ With each breath, imagine roots growing down")
        st.markdown("6. ✅ Focus on present moment")
        st.markdown("7. ✅ Continue until feeling grounded")
        
    elif emotion == "sad":
        st.info("**Comforting Breath** - Gentle breathing for emotional support")
        st.markdown("**Scientific Reference:** [Neff (2011) - Self-Compassion](https://self-compassion.org/)")
        
        st.markdown("**Practice Steps:**")
        st.markdown("1. ✅ Lie down or sit comfortably")
        st.markdown("2. ✅ Place hand on heart")
        st.markdown("3. ✅ Breathe softly and slowly")
        st.markdown("4. ✅ With each breath, offer yourself kindness")
        st.markdown("5. ✅ Imagine warm, healing light")
        st.markdown("6. ✅ Allow emotions to flow naturally")
        st.markdown("7. ✅ Practice self-acceptance")
        
    elif emotion == "happy":
        st.info("**Joy Amplification Breath** - Energizing breathing to enhance positive emotions")
        st.markdown("**Scientific Reference:** [Kok et al. (2013) - Positive Emotions](https://pubmed.ncbi.nlm.nih.gov/23527591/)")
        
        st.markdown("**Practice Steps:**")
        st.markdown("1. ✅ Stand or sit with open posture")
        st.markdown("2. ✅ Smile gently")
        st.markdown("3. ✅ Breathe in joy and gratitude")
        st.markdown("4. ✅ Exhale spreading happiness")
        st.markdown("5. ✅ Visualize golden light filling your body")
        st.markdown("6. ✅ Feel energy expanding outward")
        st.markdown("7. ✅ Share joy with the world")
        
    else:
        st.info("**Mindful Breathing** - Present-moment awareness breathing")
        st.markdown("**Scientific Reference:** [Kabat-Zinn (1990) - Mindfulness](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3679190/)")
        
        st.markdown("**Practice Steps:**")
        st.markdown("1. ✅ Sit comfortably")
        st.markdown("2. ✅ Close eyes gently")
        st.markdown("3. ✅ Notice natural breath")
        st.markdown("4. ✅ Don't change breathing")
        st.markdown("5. ✅ Observe sensations")
        st.markdown("6. ✅ When mind wanders, return to breath")
        st.markdown("7. ✅ Practice non-judgmental awareness")
    
    st.markdown("**Why This Exercise Helps:**")
    st.markdown(f"This breathing technique specifically targets the physiological and psychological aspects of **{emotion}** emotion. The scientific research shows that controlled breathing activates the parasympathetic nervous system, reduces stress hormones, and promotes emotional regulation.")

def clean_text_for_pdf(text: str) -> str:
    """Clean text to remove Unicode characters that can't be encoded in latin-1"""
    if not text:
        return ""
    # Replace common Unicode characters with ASCII equivalents
    replacements = {
        '–': '-',  # en dash to hyphen
        '—': '-',  # em dash to hyphen
        '"': '"',  # left double quotation mark
        '"': '"',  # right double quotation mark
        ''': "'",  # left single quotation mark
        ''': "'",  # right single quotation mark
        '…': '...',  # horizontal ellipsis
        '•': '*',  # bullet point
    }
    
    cleaned_text = text
    for unicode_char, ascii_char in replacements.items():
        cleaned_text = cleaned_text.replace(unicode_char, ascii_char)
    
    return cleaned_text

def build_pdf(session_info: Dict, percentages: Dict[str, float], top_emotion: str, recs, detection_images: List = None):
    songs, reads, therapy = recs
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "AI MoodMate - Session Summary", ln=True)

    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, f"Date/Time: {session_info['timestamp']}", ln=True)
    pdf.cell(0, 8, f"Input Mode: {session_info['input_mode']}", ln=True)
    pdf.ln(4)

    # Detection Images
    if detection_images and len(detection_images) > 0:
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, "Detection Images:", ln=True)
        pdf.set_font("Arial", size=10)
        
        for i, img in enumerate(detection_images[:3]):  # Show max 3 images
            try:
                # Save image temporarily
                img_path = os.path.join(OUTPUTS_DIR, f"temp_detection_{i}_{int(time.time())}.png")
                cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                
                # Add image to PDF
                pdf.image(img_path, w=80, h=60)
                pdf.cell(0, 5, f"Detection Image {i+1}", ln=True)
                pdf.ln(2)
                
                # Clean up temp file
                if os.path.exists(img_path):
                    os.remove(img_path)
            except Exception as e:
                pdf.cell(0, 5, f"Image {i+1}: Error loading image", ln=True)
        
        pdf.ln(4)

    # Percentages
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Average Emotion Percentages:", ln=True)
    pdf.set_font("Arial", size=11)
    for k in EMOTION_CLASSES:
        pdf.cell(0, 7, f"- {k.capitalize()}: {percentages.get(k, 0.0)}%", ln=True)

    pdf.ln(4)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, f"Dominant Emotion: {top_emotion.capitalize()}", ln=True)

    # Songs
    pdf.ln(4)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Recommended Songs:", ln=True)
    pdf.set_font("Arial", size=10)
    for title, reason, link in songs:
        clean_title = clean_text_for_pdf(title)
        clean_reason = clean_text_for_pdf(reason)
        clean_link = clean_text_for_pdf(link)
        # Split into multiple lines to avoid text overflow
        pdf.set_font("Arial", "B", 10)
        pdf.multi_cell(0, 5, f"- {clean_title}")
        pdf.set_font("Arial", size=9)
        pdf.multi_cell(0, 4, f"  Reason: {clean_reason}")
        pdf.multi_cell(0, 4, f"  Link: {clean_link}")
        pdf.ln(1)

    # Reading/Mindfulness
    pdf.ln(2)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Reading & Mindfulness:", ln=True)
    pdf.set_font("Arial", size=10)
    for title, reason, link in reads:
        clean_title = clean_text_for_pdf(title)
        clean_reason = clean_text_for_pdf(reason)
        clean_link = clean_text_for_pdf(link)
        # Split into multiple lines to avoid text overflow
        pdf.set_font("Arial", "B", 10)
        pdf.multi_cell(0, 5, f"- {clean_title}")
        pdf.set_font("Arial", size=9)
        pdf.multi_cell(0, 4, f"  Why: {clean_reason}")
        pdf.multi_cell(0, 4, f"  Link: {clean_link}")
        pdf.ln(1)

    # Therapy
    pdf.ln(2)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Support & Counseling Resources:", ln=True)
    pdf.set_font("Arial", size=10)
    for name, desc, link in therapy:
        clean_name = clean_text_for_pdf(name)
        clean_desc = clean_text_for_pdf(desc)
        clean_link = clean_text_for_pdf(link)
        # Split into multiple lines to avoid text overflow
        pdf.set_font("Arial", "B", 10)
        pdf.multi_cell(0, 5, f"- {clean_name}")
        pdf.set_font("Arial", size=9)
        pdf.multi_cell(0, 4, f"  {clean_desc}")
        pdf.multi_cell(0, 4, f"  Link: {clean_link}")
        pdf.ln(1)

    # Save
    filename = f"moodmate_summary_{int(time.time())}.pdf"
    pdf_path = os.path.join(OUTPUTS_DIR, filename)
    pdf.output(pdf_path)
    return pdf_path

# ----------------------------
# Streamlit UI
# ----------------------------

# Professional Header
st.markdown("""
<div class="header-container">
    <div class="app-title">🧠 AI MoodMate</div>
    <div class="app-subtitle">Facial Emotion Detection → Mood Insights → Music & Wellness Recommendations</div>
</div>
""", unsafe_allow_html=True)

# Initialize session data
initialize_session_data()

# Loading indicator
with st.spinner("Loading AI Model..."):
    model = load_model()
st.success("✅ AI Model Loaded Successfully!")

# Enhanced Sidebar with Mood Dashboard
st.markdown("""
<div class="sidebar">
""", unsafe_allow_html=True)

# Mood History Dashboard
st.sidebar.markdown("### 📊 Mood Dashboard")
insights = get_mood_insights()

if insights:
    st.sidebar.metric("Total Sessions", insights['total_sessions'])
    st.sidebar.metric("Most Common Mood", insights['most_common_emotion'].capitalize())
    
    # Mood stability indicator
    stability_color = "🟢" if insights['mood_stability'] > 0.6 else "🟡" if insights['mood_stability'] > 0.3 else "🔴"
    st.sidebar.markdown(f"**Mood Stability:** {stability_color} {insights['mood_stability']:.1%}")
    
    # Recent trend
    if insights['recent_trend']:
        st.sidebar.markdown("**Recent Trend:**")
        for emotion, count in list(insights['recent_trend'].items())[:3]:
            st.sidebar.markdown(f"• {emotion.capitalize()}: {count} times")
else:
    st.sidebar.markdown("**Welcome!** Start your first session to see your mood insights.")

st.sidebar.markdown("---")

st.sidebar.markdown("### 🎯 Input Mode")
mode = st.sidebar.radio("Choose one", ["Image", "Video", "Live Webcam", "Text Input"], key="input_mode")

st.sidebar.markdown("### ⚙️ Settings")
conf_thr = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.25, 0.05, key="confidence")

run_inference = st.sidebar.button("🚀 Run Detection", type="primary", use_container_width=True)

# Quick access to mood history
if st.sidebar.button("📈 View Full History", use_container_width=True):
    st.session_state.show_history = True

st.markdown("""
</div>
""", unsafe_allow_html=True)

# Show full mood history if requested
if st.session_state.get('show_history', False):
    st.markdown("""
    <div class="card">
        <h3>📈 Complete Mood History</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.mood_history:
        # Create a detailed DataFrame for better visualization
        history_data = []
        for session in st.session_state.mood_history:
            # Get emotion percentages for display
            emotion_percentages = session.get('emotion_percentages', {})
            top_emotions = sorted(emotion_percentages.items(), key=lambda x: x[1], reverse=True)[:3]
            emotion_summary = ", ".join([f"{emotion.capitalize()}({percent:.1f}%)" for emotion, percent in top_emotions])
            
            history_data.append({
                'Date': session['date'],
                'Time': session['time'],
                'Input Mode': session['input_mode'],
                'Dominant Emotion': session['dominant_emotion'].capitalize(),
                'Emotion Breakdown': emotion_summary,
                'Session ID': session['session_id']
            })
        
        df_history = pd.DataFrame(history_data)
        st.dataframe(df_history, width='stretch')
        
        # Show session statistics
        st.markdown("### 📊 Session Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Sessions", len(st.session_state.mood_history))
        
        with col2:
            mode_counts = {}
            for session in st.session_state.mood_history:
                mode = session['input_mode']
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
            most_used_mode = max(mode_counts.items(), key=lambda x: x[1])[0] if mode_counts else "None"
            st.metric("Most Used Mode", most_used_mode)
        
        with col3:
            emotion_counts = {}
            for session in st.session_state.mood_history:
                emotion = session['dominant_emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            most_common_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "None"
            st.metric("Most Common Emotion", most_common_emotion.capitalize())
        
        with col4:
            if len(st.session_state.mood_history) > 1:
                first_session = st.session_state.mood_history[0]['timestamp']
                last_session = st.session_state.mood_history[-1]['timestamp']
                days_active = (last_session - first_session).days + 1
                st.metric("Days Active", days_active)
            else:
                st.metric("Days Active", 1)
        
        # Mood trend chart
        if len(st.session_state.mood_history) > 1:
            st.markdown("### 📊 Mood Trend Over Time")
            emotion_counts = {}
            for session in st.session_state.mood_history:
                emotion = session['dominant_emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            trend_df = pd.DataFrame(list(emotion_counts.items()), columns=['Emotion', 'Count'])
            fig_trend = px.pie(trend_df, values='Count', names='Emotion', title="Overall Mood Distribution")
            st.plotly_chart(fig_trend, width='stretch')
        
        # Clear history option
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.mood_history = []
            st.session_state.user_preferences['session_count'] = 0
            st.success("History cleared!")
            st.rerun()
    else:
        st.info("No mood history available yet. Start your first session!")
    
    if st.button("← Back to Main App"):
        st.session_state.show_history = False
        st.rerun()
    
    st.stop()  # Stop execution to show only history

# Aggregation store
weights = {k: 0.0 for k in EMOTION_CLASSES}
detection_images = []  # Store detection images for PDF

# ----------------------------
# IMAGE MODE
# ----------------------------
if mode == "Image":
    st.markdown("""
    <div class="card">
        <h3>📸 Image Input</h3>
    </div>
    """, unsafe_allow_html=True)
    
    file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"], help="Upload a clear image with visible faces")
    
    if run_inference and file is not None:
        with st.spinner("🔍 Analyzing emotions..."):
            progress_bar = st.progress(0)
            progress_bar.progress(25)
            
            img = Image.open(file).convert("RGB")
            img_np = np.array(img)
            progress_bar.progress(50)
            
            res = model.predict(img_np, conf=conf_thr, verbose=False)
            progress_bar.progress(75)
            
            out_img = draw_detections(img_np, res, conf_thr)
            progress_bar.progress(100)

        accumulate_emotions(res, weights)
        percentages = normalize_percentages(weights)
        
        # Store detection image for PDF
        detection_images = [out_img]

        # Results Display
        st.markdown("""
        <div class="card">
            <h3>🎯 Detection Results</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(out_img, caption="Detection Result", width='stretch')
        with col2:
            st.markdown("""
            <div class="chart-container">
            """, unsafe_allow_html=True)
            df = pd.DataFrame({"Emotion": list(percentages.keys()), "Percentage": list(percentages.values())})
            fig = px.bar(df, x="Emotion", y="Percentage", title="Emotion Distribution", 
                        color="Emotion", color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig, width='stretch')
            st.markdown("</div>", unsafe_allow_html=True)

        dom = dominant_emotion(percentages)
        st.markdown(f"""
        <div class="success-message">
            <h4>🎉 Dominant Emotion: {dom.capitalize()}</h4>
        </div>
        """, unsafe_allow_html=True)

        # Get Personalized Recommendations
        songs, reads, therapy, breathing = get_personalized_recommendations(dom)
        
        # Save session data
        emotion_data = {'dominant': dom, 'percentages': percentages}
        recommendations = (songs, reads, therapy, breathing)
        save_mood_session(emotion_data, recommendations, "Image")
        
        st.markdown("""
        <div class="recommendation-section">
            <h3>🎵 Music Picks</h3>
        </div>
        """, unsafe_allow_html=True)
        for t, r, link in songs:
            st.markdown(f"- [{t}]({link}) — _{r}_")

        st.markdown("""
        <div class="recommendation-section">
            <h3>📚 Reading & Mindfulness</h3>
        </div>
        """, unsafe_allow_html=True)
        for t, r, link in reads:
            st.markdown(f"- [{t}]({link}) — _{r}_")

        st.markdown("""
        <div class="recommendation-section">
            <h3>🧑‍⚕️ Therapy & Counseling</h3>
        </div>
        """, unsafe_allow_html=True)
        for n, d, link in therapy:
            st.markdown(f"- [{n}]({link}) — _{d}_")

        # Breathing Exercise
        display_breathing_exercise(dom)

        # PDF Download
        st.markdown("""
        <div class="card">
            <h3>📄 Session Summary</h3>
        </div>
        """, unsafe_allow_html=True)
        
        session_info = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "input_mode": "Image"}
        pdf_path = build_pdf(session_info, percentages, dom, (songs, reads, therapy), detection_images)
        with open(pdf_path, "rb") as f:
            st.download_button("📥 Download Session PDF", f, file_name=os.path.basename(pdf_path), 
                            mime="application/pdf", type="primary", use_container_width=True)

# ----------------------------
# VIDEO MODE
# ----------------------------
elif mode == "Video":
    st.subheader("Video Input")
    vfile = st.file_uploader("Upload a video", type=["mp4","mov","avi","mkv"])
    if run_inference and vfile is not None:
        # Read video bytes into temp buffer
        tname = os.path.join(OUTPUTS_DIR, f"temp_{int(time.time())}.mp4")
        with open(tname, "wb") as f:
            f.write(vfile.read())

        cap = cv2.VideoCapture(tname)
        frame_count = 0
        preview_every = max(1, int(cap.get(cv2.CAP_PROP_FPS)) // 3)
        detection_images = []  # Store detection images for PDF

        preview_placeholder = st.empty()
        progress = st.progress(0)

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_count += 1
            frame_rgb = bgr_to_rgb(frame_bgr)

            res = model.predict(frame_rgb, conf=conf_thr, verbose=False)
            accumulate_emotions(res, weights)

            if frame_count % preview_every == 0:
                out = draw_detections(frame_rgb, res, conf_thr)
                preview_placeholder.image(out, caption=f"Frame {frame_count}", width='stretch')
                # Store sample detection images for PDF (max 5)
                if len(detection_images) < 5:
                    detection_images.append(out)

            # update progress
            if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
                progress_val = min(1.0, frame_count / cap.get(cv2.CAP_PROP_FRAME_COUNT))
                progress.progress(progress_val)

        cap.release()

        percentages = normalize_percentages(weights)
        df = pd.DataFrame({"Emotion": list(percentages.keys()), "Percentage": list(percentages.values())})
        fig = px.bar(df, x="Emotion", y="Percentage", title="Average Emotion Percentages (Video)")
        st.plotly_chart(fig, width='stretch')

        dom = dominant_emotion(percentages)
        st.markdown(f"""
        <div class="success-message">
            <h4>🎉 Dominant Emotion: {dom.capitalize()}</h4>
        </div>
        """, unsafe_allow_html=True)

        # Get Personalized Recommendations
        songs, reads, therapy, breathing = get_personalized_recommendations(dom)
        
        # Save session data
        emotion_data = {'dominant': dom, 'percentages': percentages}
        recommendations = (songs, reads, therapy, breathing)
        save_mood_session(emotion_data, recommendations, "Video")

        # View Detections feature - Auto display after processing
        if detection_images:
            st.markdown("### 📸 Detection Frames")
            st.info(f"Showing {len(detection_images)} sample detection frames")
            
            # Automatically display detection frames
            st.markdown("**Sample Detection Frames:**")
            cols = st.columns(min(3, len(detection_images)))
            for i, det_img in enumerate(detection_images):
                with cols[i % 3]:
                    st.image(det_img, caption=f"Detection Frame {i+1}", width='stretch')
        else:
            st.warning("No detection images found!")

        songs, reads, therapy, breathing = recommend_content(dom)
        st.markdown("### 🎵 Music Picks (click to open)")
        for t, r, link in songs:
            st.markdown(f"- [{t}]({link}) — _{r}_")

        st.markdown("### 📚 Reading & Mindfulness")
        for t, r, link in reads:
            st.markdown(f"- [{t}]({link}) — _{r}_")

        st.markdown("### 🧑‍⚕️ Therapy & Counseling")
        for n, d, link in therapy:
            st.markdown(f"- [{n}]({link}) — _{d}_")

        # Breathing Exercise
        display_breathing_exercise(dom)

        session_info = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "input_mode": "Video"}
        pdf_path = build_pdf(session_info, percentages, dom, (songs, reads, therapy), detection_images)
        with open(pdf_path, "rb") as f:
            st.download_button("Download Session PDF", f, file_name=os.path.basename(pdf_path), mime="application/pdf")

# ----------------------------
# LIVE WEBCAM MODE
# ----------------------------
elif mode == "Live Webcam":
    st.subheader("Live Webcam Input")

    class EmotionTransformer(VideoTransformerBase):
        def __init__(self):
            self.model = model
            self.conf = conf_thr
            self.weights = {k: 0.0 for k in EMOTION_CLASSES}
            self.detection_images = []
            self.frame_count = 0

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = self.model.predict(rgb, conf=self.conf, verbose=False)
            accumulate_emotions(res, self.weights)
            drawn = draw_detections(rgb, res, self.conf)
            
            # Store sample detection images (max 5) - every 10th frame
            self.frame_count += 1
            if self.frame_count % 10 == 0 and len(self.detection_images) < 5:
                self.detection_images.append(drawn.copy())
            
            out_bgr = cv2.cvtColor(drawn, cv2.COLOR_RGB2BGR)
            return av.VideoFrame.from_ndarray(out_bgr, format="bgr24")

    ctx = webrtc_streamer(
        key="moodmate",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=EmotionTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )

    if run_inference and ctx and ctx.video_transformer:
        # capture a short dwell for aggregation
        st.info("Aggregating a few seconds of webcam frames...")
        time.sleep(2.0)  # short dwell to collect frames

        percentages = normalize_percentages(ctx.video_transformer.weights)
        df = pd.DataFrame({"Emotion": list(percentages.keys()), "Percentage": list(percentages.values())})
        fig = px.bar(df, x="Emotion", y="Percentage", title="Average Emotion Percentages (Webcam)")
        st.plotly_chart(fig, width='stretch')

        dom = dominant_emotion(percentages)
        st.markdown(f"""
        <div class="success-message">
            <h4>🎉 Dominant Emotion: {dom.capitalize()}</h4>
        </div>
        """, unsafe_allow_html=True)

        # Get Personalized Recommendations
        songs, reads, therapy, breathing = get_personalized_recommendations(dom)
        
        # Save session data
        emotion_data = {'dominant': dom, 'percentages': percentages}
        recommendations = (songs, reads, therapy, breathing)
        save_mood_session(emotion_data, recommendations, "Live Webcam")

        # View Detections feature for webcam - Auto display after processing
        if hasattr(ctx.video_transformer, 'detection_images') and ctx.video_transformer.detection_images:
            st.markdown("### 📸 Detection Frames")
            st.info(f"Showing {len(ctx.video_transformer.detection_images)} sample detection frames")
            
            # Automatically display detection frames
            st.markdown("**Sample Detection Frames:**")
            cols = st.columns(min(3, len(ctx.video_transformer.detection_images)))
            for i, det_img in enumerate(ctx.video_transformer.detection_images):
                with cols[i % 3]:
                    st.image(det_img, caption=f"Detection Frame {i+1}", width='stretch')
        else:
            st.warning("No webcam detection images found!")

        songs, reads, therapy, breathing = recommend_content(dom)
        st.markdown("### 🎵 Music Picks (click to open)")
        for t, r, link in songs:
            st.markdown(f"- [{t}]({link}) — _{r}_")

        st.markdown("### 📚 Reading & Mindfulness")
        for t, r, link in reads:
            st.markdown(f"- [{t}]({link}) — _{r}_")

        st.markdown("### 🧑‍⚕️ Therapy & Counseling")
        for n, d, link in therapy:
            st.markdown(f"- [{n}]({link}) — _{d}_")

        # Breathing Exercise
        display_breathing_exercise(dom)

        session_info = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "input_mode": "Live Webcam"}
        webcam_images = ctx.video_transformer.detection_images if hasattr(ctx.video_transformer, 'detection_images') else []
        pdf_path = build_pdf(session_info, percentages, dom, (songs, reads, therapy), webcam_images)
        with open(pdf_path, "rb") as f:
            st.download_button("Download Session PDF", f, file_name=os.path.basename(pdf_path), mime="application/pdf")

# ----------------------------
# TEXT INPUT MODE
# ----------------------------
elif mode == "Text Input":
    st.markdown("""
    <div class="card">
        <h3>📝 Text Input - Select Your Emotion</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("Choose the emotion you're currently feeling to get personalized recommendations")
    
    # Emotion selection
    selected_emotion = st.selectbox(
        "How are you feeling right now?",
        ["Select an emotion..."] + EMOTION_CLASSES,
        key="emotion_selector"
    )
    
    if selected_emotion != "Select an emotion..." and run_inference:
        # Create a simple percentage distribution (100% for selected emotion)
        percentages = {emotion: 0.0 for emotion in EMOTION_CLASSES}
        percentages[selected_emotion] = 100.0
        
        # Display emotion chart
        df = pd.DataFrame({"Emotion": list(percentages.keys()), "Percentage": list(percentages.values())})
        fig = px.bar(df, x="Emotion", y="Percentage", title="Selected Emotion", 
                    color="Emotion", color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, width='stretch')
        
        st.markdown(f"""
        <div class="success-message">
            <h4>🎯 Selected Emotion: {selected_emotion.capitalize()}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Get Personalized Recommendations
        songs, reads, therapy, breathing = get_personalized_recommendations(selected_emotion)
        
        # Save session data
        emotion_data = {'dominant': selected_emotion, 'percentages': percentages}
        recommendations = (songs, reads, therapy, breathing)
        save_mood_session(emotion_data, recommendations, "Text Input")
        
        # Display recommendations with enhanced styling
        st.markdown("""
        <div class="recommendation-section">
            <h3>🎵 Music Picks</h3>
        </div>
        """, unsafe_allow_html=True)
        for t, r, link in songs:
            st.markdown(f"- [{t}]({link}) — _{r}_")

        st.markdown("""
        <div class="recommendation-section">
            <h3>📚 Reading & Mindfulness</h3>
        </div>
        """, unsafe_allow_html=True)
        for t, r, link in reads:
            st.markdown(f"- [{t}]({link}) — _{r}_")

        st.markdown("""
        <div class="recommendation-section">
            <h3>🧑‍⚕️ Therapy & Counseling</h3>
        </div>
        """, unsafe_allow_html=True)
        for n, d, link in therapy:
            st.markdown(f"- [{n}]({link}) — _{d}_")

        display_breathing_exercise(selected_emotion)

        # PDF Download
        st.markdown("""
        <div class="card">
            <h3>📄 Session Summary</h3>
        </div>
        """, unsafe_allow_html=True)
        
        session_info = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "input_mode": "Text Input"}
        pdf_path = build_pdf(session_info, percentages, selected_emotion, (songs, reads, therapy), [])
        with open(pdf_path, "rb") as f:
            st.download_button("📥 Download Session PDF", f, file_name=os.path.basename(pdf_path), 
                            mime="application/pdf", type="primary", use_container_width=True)
    
    elif selected_emotion == "Select an emotion...":
        st.warning("Please select an emotion to get recommendations")

# Footer
st.markdown("---")
# Professional Footer
st.markdown("""
<div class="footer">
    <p><strong>🧠 AI MoodMate</strong> · Emotion detection powered by YOLOv11</p>
    <p>Music & wellbeing suggestions are supportive, not clinical advice.</p>
    <p>Built with ❤️ for emotional wellness</p>
</div>
""", unsafe_allow_html=True)
