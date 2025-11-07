#!/usr/bin/env python3
"""
Mental Health Companion - Flask Backend
Production-ready API with HuggingFace model integration and empathetic responses
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import requests
import json
import os
import logging
import time
from datetime import datetime, timedelta
from functools import wraps
import traceback
import sqlite3
import hashlib
import secrets
from werkzeug.exceptions import BadRequest, InternalServerError

# Optional Gemini AI import
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

# Audio generation import
try:
    import pyttsx3
    import uuid
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    pyttsx3 = None
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mental_health_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Enable CORS for React Native
CORS(app, origins=["*"])

# Rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)
limiter.init_app(app)

class HuggingFaceModelClient:
    """Client for interacting with HuggingFace models"""
    
    def __init__(self):
        self.username = "Utkarsh64"
        self.models = {
            'logistic_regression': f"{self.username}/mental-health-logistic-regression",
            'enhanced_distilbert': f"{self.username}/huggingface_distilbert_model"
        }
        
        # Use new HuggingFace router endpoint (updated Nov 2025)
        self.api_endpoint = "https://api-inference.huggingface.co/models"
        
        # Only add auth if token exists
        hf_token = os.environ.get('HUGGINGFACE_TOKEN', '')
        if hf_token:
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {hf_token}"
            }
        else:
            # Use public API without auth
            self.headers = {"Content-Type": "application/json"}
        
        logger.info(f"HuggingFace client initialized for user: {self.username}")
    
    def predict_emotions(self, text: str, model_type: str = 'enhanced_distilbert', timeout: int = 30) -> Dict[str, Any]:
        """Predict emotions using HuggingFace model"""
        
        if model_type not in self.models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_name = self.models[model_type]
        
        payload = {
            "inputs": text,
            "options": {
                "wait_for_model": True,
                "use_cache": False
            }
        }
        
        # Try HuggingFace API
        api_url = f"{self.api_endpoint}/{model_name}"
        
        try:
            logger.info(f"Calling HuggingFace API: {api_url}")
            
            response = requests.post(
                api_url, 
                headers=self.headers, 
                json=payload, 
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"HuggingFace API success: {model_type}")
                return self._process_hf_response(result, model_type)
            
            elif response.status_code == 503:
                logger.warning(f"Model loading (503), using fallback")
                return self._fallback_emotion_detection(text, model_type)
            
            else:
                logger.error(f"HuggingFace API error {response.status_code}: {response.text}")
                return self._fallback_emotion_detection(text, model_type)
                
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout for {api_url}, using fallback")
            return self._fallback_emotion_detection(text, model_type)
        except Exception as e:
            logger.error(f"Request error for {api_url}: {e}")
            return self._fallback_emotion_detection(text, model_type)
        
        # If all APIs fail, return fallback response
        logger.warning(f"All HuggingFace APIs failed for {model_type}, using fallback")
        return self._fallback_emotion_detection(text, model_type)
    
    def _process_hf_response(self, result: Any, model_type: str) -> Dict[str, Any]:
        """Process HuggingFace API response"""
        
        try:
            if isinstance(result, list) and len(result) > 0:
                # Standard classification response
                emotions = []
                for item in result:
                    if isinstance(item, dict) and 'label' in item and 'score' in item:
                        emotions.append({
                            'emotion': item['label'],
                            'confidence': float(item['score']),
                            'predicted': float(item['score']) > 0.4
                        })
                
                # Sort by confidence
                emotions.sort(key=lambda x: x['confidence'], reverse=True)
                
                return {
                    'success': True,
                    'model_used': model_type,
                    'emotions': emotions,
                    'top_emotions': emotions[:5],
                    'predicted_emotions': [e for e in emotions if e['predicted']],
                    'confidence_score': emotions[0]['confidence'] if emotions else 0.0,
                    'source': 'huggingface_api'
                }
            
            else:
                logger.warning(f"Unexpected HuggingFace response format: {result}")
                return self._fallback_emotion_detection("", model_type)
                
        except Exception as e:
            logger.error(f"Error processing HuggingFace response: {e}")
            return self._fallback_emotion_detection("", model_type)
    
    def _fallback_emotion_detection(self, text: str, model_type: str) -> Dict[str, Any]:
        """Fallback emotion detection when HuggingFace APIs are unavailable"""
        
        # Enhanced keyword-based emotion detection as fallback
        emotion_keywords = {
            'joy': ['happy', 'excited', 'great', 'amazing', 'wonderful', 'fantastic', 'good', 'glad', 'pleased', 'delighted', 'joyful'],
            'sadness': ['sad', 'depressed', 'down', 'upset', 'disappointed', 'hurt', 'unhappy', 'miserable', 'heartbroken', 'lonely'],
            'anger': ['angry', 'mad', 'furious', 'annoyed', 'frustrated', 'irritated', 'rage', 'hate', 'pissed'],
            'fear': ['scared', 'afraid', 'anxious', 'worried', 'nervous', 'terrified', 'panic', 'stress', 'overwhelmed'],
            'love': ['love', 'adore', 'cherish', 'care', 'affection', 'fond', 'devoted'],
            'gratitude': ['thank', 'grateful', 'appreciate', 'thankful', 'blessed'],
            'confusion': ['confused', 'puzzled', 'unclear', 'don\'t understand', 'lost', 'uncertain'],
            'excitement': ['excited', 'thrilled', 'pumped', 'enthusiastic', 'eager', 'energized'],
            'optimism': ['hope', 'hopeful', 'optimistic', 'positive', 'confident', 'looking forward'],
            'disappointment': ['disappointed', 'let down', 'discouraged', 'dismayed'],
            'relief': ['relief', 'relieved', 'calm', 'relaxed', 'peaceful'],
            'pride': ['proud', 'accomplished', 'achieved', 'success'],
            'caring': ['care', 'concern', 'support', 'help', 'compassion'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'unexpected']
        }
        
        text_lower = text.lower()
        detected_emotions = []
        
        for emotion, keywords in emotion_keywords.items():
            score = 0.0
            matches = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 0.25
                    matches += 1
            
            if score > 0:
                # Boost score for multiple keyword matches
                if matches > 1:
                    score += 0.2
                
                detected_emotions.append({
                    'emotion': emotion,
                    'confidence': min(score, 0.95),
                    'predicted': True
                })
        
        # If no emotions detected, analyze sentiment
        if not detected_emotions:
            # Check for positive/negative words
            positive_words = ['good', 'nice', 'fine', 'okay', 'well']
            negative_words = ['bad', 'not', 'no', 'never', 'nothing']
            
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                detected_emotions.append({'emotion': 'optimism', 'confidence': 0.6, 'predicted': True})
            elif neg_count > pos_count:
                detected_emotions.append({'emotion': 'sadness', 'confidence': 0.5, 'predicted': True})
            else:
                detected_emotions.append({'emotion': 'neutral', 'confidence': 0.7, 'predicted': True})
        
        # Sort by confidence
        detected_emotions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Ensure we always return at least 5 emotions for display
        while len(detected_emotions) < 5:
            remaining_emotions = ['caring', 'curiosity', 'realization', 'approval', 'admiration']
            for em in remaining_emotions:
                if em not in [e['emotion'] for e in detected_emotions]:
                    detected_emotions.append({'emotion': em, 'confidence': 0.3, 'predicted': False})
                    if len(detected_emotions) >= 5:
                        break
        
        return {
            'success': True,
            'model_used': f"{model_type}_fallback",
            'emotions': detected_emotions[:10],
            'top_emotions': detected_emotions[:5],
            'predicted_emotions': [e for e in detected_emotions if e['predicted']][:5],
            'confidence_score': detected_emotions[0]['confidence'] if detected_emotions else 0.0,
            'source': 'fallback_keywords'
        }

class EmpathyEngine:
    """Generate empathetic responses based on detected emotions"""
    
    def __init__(self):
        # Initialize Gemini API if available
        self.gemini_available = False
        try:
            if GEMINI_AVAILABLE:
                gemini_key = os.environ.get('GEMINI_API_KEY')
                if gemini_key:
                    genai.configure(api_key=gemini_key)
                    
                    # Try different model names (don't test during init, test on first use)
                    model_names = ['gemini-pro', 'gemini-1.5-flash', 'gemini-1.5-pro']
                    
                    for model_name in model_names:
                        try:
                            logger.info(f"🔍 Initializing Gemini model: {model_name}")
                            self.gemini_model = genai.GenerativeModel(model_name)
                            self.gemini_available = True
                            self.gemini_model_name = model_name
                            logger.info(f"✅ Gemini model {model_name} initialized (will test on first use)")
                            break
                        except Exception as e:
                            logger.warning(f"⚠️ Model {model_name} initialization failed: {e}")
                            continue
                    
                    if not self.gemini_available:
                        logger.error("❌ All Gemini models failed to initialize")
                        logger.error("Check your GEMINI_API_KEY and internet connection")
                else:
                    logger.warning("⚠️ GEMINI_API_KEY not set in environment variables")
                    logger.warning("⚠️ Set GEMINI_API_KEY in Render dashboard to enable AI responses")
            else:
                logger.warning("⚠️ google-generativeai library not installed")
                logger.warning("⚠️ Run: pip install google-generativeai")
        except Exception as e:
            logger.error(f"❌ Gemini API initialization failed: {e}")
            logger.error(f"Error details: {traceback.format_exc()}")
        
        # Load response templates
        self.response_templates = self._load_response_templates()
        logger.info("Empathy engine initialized")
    
    def _load_response_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Load empathetic response templates"""
        
        return {
            'joy': {
                'acknowledgment': [
                    "I can feel the happiness in your message!",
                    "It's wonderful to hear such positive energy from you!",
                    "Your joy is really coming through!"
                ],
                'validation': [
                    "It's beautiful to experience such positive emotions.",
                    "Moments of joy like this are precious.",
                    "You deserve to feel this happiness."
                ],
                'support': [
                    "I'm so glad you're experiencing this joy. What's bringing you such happiness?",
                    "It's lovely to share in your positive moment. Tell me more about what's making you feel so good!",
                    "Your happiness is contagious! What's the source of this wonderful feeling?"
                ]
            },
            'sadness': {
                'acknowledgment': [
                    "I can sense that you're going through a difficult time right now.",
                    "I hear the sadness in your words, and I want you to know I'm here.",
                    "It sounds like you're carrying some heavy feelings."
                ],
                'validation': [
                    "It's completely okay to feel sad. Your emotions are valid.",
                    "Sadness is a natural part of the human experience.",
                    "You don't have to hide or rush through these feelings."
                ],
                'support': [
                    "I'm here to listen without judgment. Would you like to share what's weighing on your heart?",
                    "Sometimes talking about what's making us sad can help. I'm here if you want to open up.",
                    "You don't have to go through this alone. What's been troubling you?"
                ]
            },
            'anger': {
                'acknowledgment': [
                    "I can tell you're feeling really frustrated or angry right now.",
                    "It sounds like something has really upset you.",
                    "I hear the intensity of your feelings."
                ],
                'validation': [
                    "Your anger is valid - something has clearly affected you deeply.",
                    "It's natural to feel angry when things don't go as expected.",
                    "These feelings of frustration make complete sense."
                ],
                'support': [
                    "Would you like to talk about what's making you feel this way?",
                    "Sometimes expressing anger in a safe space can be helpful. I'm here to listen.",
                    "What's been building up that's led to these feelings?"
                ]
            },
            'fear': {
                'acknowledgment': [
                    "I can sense that you're feeling anxious or scared about something.",
                    "It sounds like you're dealing with some worrying thoughts.",
                    "I hear the concern and fear in what you're sharing."
                ],
                'validation': [
                    "Fear and anxiety are natural responses to uncertainty.",
                    "It's completely understandable to feel scared sometimes.",
                    "Your worries are real and valid."
                ],
                'support': [
                    "You're not alone in feeling this way. What's been causing you to feel anxious?",
                    "Sometimes sharing our fears can make them feel less overwhelming. What's on your mind?",
                    "I'm here to support you through this. What's been worrying you?"
                ]
            },
            'confusion': {
                'acknowledgment': [
                    "It sounds like you're feeling uncertain or confused about something.",
                    "I can sense that things might feel unclear right now.",
                    "It seems like you're trying to make sense of something difficult."
                ],
                'validation': [
                    "Confusion is a normal part of processing complex situations.",
                    "It's okay not to have all the answers right now.",
                    "Feeling uncertain doesn't mean you're doing anything wrong."
                ],
                'support': [
                    "Would it help to talk through what's been confusing you?",
                    "Sometimes discussing our thoughts can bring clarity. What's been on your mind?",
                    "I'm here to help you work through whatever is feeling unclear."
                ]
            },
            'neutral': {
                'acknowledgment': [
                    "Thank you for sharing with me.",
                    "I'm here and listening to what you have to say.",
                    "I appreciate you taking the time to connect."
                ],
                'validation': [
                    "Whatever you're feeling right now is completely valid.",
                    "It's okay to have moments where emotions feel balanced.",
                    "Sometimes neutral moments are exactly what we need."
                ],
                'support': [
                    "How are you doing today? I'm here if you'd like to share anything.",
                    "Is there anything on your mind that you'd like to talk about?",
                    "I'm here to listen and support you in whatever way you need."
                ]
            }
        }
    
    def generate_response(self, text: str, emotions: Dict[str, Any], user_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate empathetic response based on emotions"""
        
        try:
            # Get primary emotions
            predicted_emotions = emotions.get('predicted_emotions', [])
            top_emotions = emotions.get('top_emotions', [])
            
            if not predicted_emotions and top_emotions:
                # Use top emotion if no predictions above threshold
                primary_emotion = top_emotions[0]['emotion']
            elif predicted_emotions:
                primary_emotion = predicted_emotions[0]['emotion']
            else:
                primary_emotion = 'neutral'
            
            # Check if we should use advanced AI response
            use_advanced = self._should_use_advanced_response(emotions, text)
            logger.info(f"Using advanced AI: {use_advanced}, Gemini available: {self.gemini_available}")
            
            if use_advanced:
                return self._generate_advanced_response(text, emotions, primary_emotion)
            else:
                return self._generate_template_response(text, emotions, primary_emotion)
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._generate_fallback_response()
    
    def _should_use_advanced_response(self, emotions: Dict[str, Any], text: str) -> bool:
        """Decide whether to use advanced AI response"""
        
        # ALWAYS use Gemini AI if available for better responses
        return self.gemini_available
    
    def _generate_advanced_response(self, text: str, emotions: Dict[str, Any], primary_emotion: str) -> Dict[str, Any]:
        """Generate response using advanced AI"""
        
        try:
            # Create context-rich prompt
            predicted_emotions = emotions.get('predicted_emotions', [])
            emotion_list = [e['emotion'] for e in predicted_emotions[:3]]
            
            prompt = f"""You are a warm, empathetic mental health companion AI. A person has just shared this with you:

"{text}"

Detected emotions: {', '.join(emotion_list) if emotion_list else primary_emotion}

Respond as a caring friend and mental health supporter. Your response should:
- Be warm, genuine, and conversational (like talking to a trusted friend)
- Acknowledge and validate their feelings without judgment
- Show deep empathy and understanding
- Offer gentle encouragement or perspective if appropriate
- Ask a thoughtful follow-up question to continue the conversation
- Be 2-4 sentences (50-100 words)
- Sound natural and human, not robotic

Important:
- Don't give medical advice or diagnose
- Don't be overly formal or clinical
- Don't use phrases like "I'm an AI" or "as an AI"
- Be supportive but not patronizing
- Match their emotional tone

Respond now:"""

            logger.info(f"🤖 Calling Gemini model: {getattr(self, 'gemini_model_name', 'unknown')}")
            
            response = self.gemini_model.generate_content(prompt)
            
            logger.info(f"✅ Gemini response received: {response.text[:50]}...")
            
            return {
                'success': True,
                'response': response.text,
                'type': 'advanced_ai',
                'primary_emotion': primary_emotion,
                'emotions_addressed': emotion_list,
                'confidence': emotions.get('confidence_score', 0.0)
            }
            
        except Exception as e:
            logger.error(f"❌ Advanced AI response failed: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Full error: {traceback.format_exc()}")
            logger.warning("⚠️ Falling back to template responses")
            return self._generate_template_response(text, emotions, primary_emotion)
    
    def _generate_template_response(self, text: str, emotions: Dict[str, Any], primary_emotion: str) -> Dict[str, Any]:
        """Generate response using templates"""
        
        import random
        
        # Get templates for primary emotion
        templates = self.response_templates.get(primary_emotion, self.response_templates['neutral'])
        
        # Build response
        response_parts = []
        
        # Acknowledgment
        response_parts.append(random.choice(templates['acknowledgment']))
        
        # Validation
        response_parts.append(random.choice(templates['validation']))
        
        # Support
        response_parts.append(random.choice(templates['support']))
        
        response_text = ' '.join(response_parts)
        
        return {
            'success': True,
            'response': response_text,
            'type': 'template_based',
            'primary_emotion': primary_emotion,
            'emotions_addressed': [primary_emotion],
            'confidence': emotions.get('confidence_score', 0.0)
        }
    
    def _generate_fallback_response(self) -> Dict[str, Any]:
        """Generate fallback response when everything fails"""
        
        return {
            'success': True,
            'response': "Thank you for sharing with me. I'm here to listen and support you. How are you feeling right now?",
            'type': 'fallback',
            'primary_emotion': 'neutral',
            'emotions_addressed': ['neutral'],
            'confidence': 0.5
        }

class AudioGenerator:
    """Generate audio responses from text using pyttsx3"""
    
    def __init__(self):
        self.audio_available = AUDIO_AVAILABLE
        self.audio_dir = 'temp_audio'
        self._ensure_audio_directory()
        
        if self.audio_available:
            try:
                # Try different TTS engines for Linux compatibility
                engines_to_try = ['espeak', 'sapi5', 'nsss', 'dummy']
                
                self.engine = None
                for engine_name in engines_to_try:
                    try:
                        self.engine = pyttsx3.init(engine_name)
                        logger.info(f"Successfully initialized TTS engine: {engine_name}")
                        break
                    except Exception as engine_error:
                        logger.warning(f"Failed to initialize {engine_name}: {engine_error}")
                        continue
                
                if self.engine is None:
                    # Try default initialization
                    self.engine = pyttsx3.init()
                
                if self.engine:
                    # Configure voice settings
                    try:
                        voices = self.engine.getProperty('voices')
                        if voices and len(voices) > 0:
                            # Try to use a female voice if available
                            for voice in voices:
                                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                                    self.engine.setProperty('voice', voice.id)
                                    break
                        
                        # Set speech rate (slower for therapeutic effect)
                        self.engine.setProperty('rate', 160)
                        
                        # Set volume
                        self.engine.setProperty('volume', 0.9)
                        
                        logger.info("Audio generator configured successfully")
                    except Exception as config_error:
                        logger.warning(f"Audio configuration failed: {config_error}")
                        # Continue anyway - basic TTS might still work
                
                logger.info("Audio generator initialized successfully")
                
            except Exception as e:
                logger.warning(f"Audio engine initialization failed: {e}")
                logger.info("Audio will be disabled - app will continue with text-only responses")
                self.audio_available = False
        else:
            logger.warning("pyttsx3 not available - audio generation disabled")
    
    def _ensure_audio_directory(self):
        """Ensure audio directory exists"""
        if not os.path.exists(self.audio_dir):
            os.makedirs(self.audio_dir)
    
    def generate_audio(self, text: str) -> Optional[str]:
        """Generate audio file from text"""
        
        if not self.audio_available or not self.engine:
            logger.info("Audio generation skipped - engine not available")
            return None
        
        try:
            # Generate unique filename
            audio_id = str(uuid.uuid4())[:8]
            filename = f"response_{audio_id}.wav"
            filepath = os.path.join(self.audio_dir, filename)
            
            # Limit text length for better performance
            if len(text) > 500:
                text = text[:500] + "..."
            
            # Audio generation disabled for now (Render compatibility)
            logger.info(f"Audio generation skipped (disabled): {text[:50]}...")
            return None
                
        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
            # Don't let audio failure break the response
            return None
    
    def cleanup_old_audio(self, max_files: int = 50):
        """Clean up old audio files to prevent storage bloat"""
        
        try:
            if not os.path.exists(self.audio_dir):
                return
            
            # Get all audio files sorted by creation time
            audio_files = []
            for filename in os.listdir(self.audio_dir):
                if filename.endswith('.wav'):
                    filepath = os.path.join(self.audio_dir, filename)
                    audio_files.append((filepath, os.path.getctime(filepath)))
            
            # Sort by creation time (oldest first)
            audio_files.sort(key=lambda x: x[1])
            
            # Remove oldest files if we exceed max_files
            if len(audio_files) > max_files:
                files_to_remove = audio_files[:-max_files]
                for filepath, _ in files_to_remove:
                    try:
                        os.remove(filepath)
                        logger.info(f"Cleaned up old audio file: {os.path.basename(filepath)}")
                    except Exception as e:
                        logger.warning(f"Failed to remove audio file {filepath}: {e}")
                        
        except Exception as e:
            logger.error(f"Audio cleanup failed: {e}")

class DatabaseManager:
    """Manage user conversations and analytics"""
    
    def __init__(self):
        self.db_path = 'mental_health_conversations.db'
        self._init_database()
        logger.info("Database manager initialized")
    
    def _init_database(self):
        """Initialize SQLite database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                message TEXT,
                emotions TEXT,
                response TEXT,
                model_used TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create analytics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE,
                total_conversations INTEGER,
                model_usage TEXT,
                emotion_distribution TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_conversation(self, user_id: str, message: str, emotions: Dict, response: Dict, model_used: str):
        """Save conversation to database"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO conversations (user_id, message, emotions, response, model_used)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                user_id,
                message,
                json.dumps(emotions),
                json.dumps(response),
                model_used
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
    
    def get_user_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get user conversation history"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT message, emotions, response, model_used, timestamp
                FROM conversations
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (user_id, limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            history = []
            for row in rows:
                history.append({
                    'message': row[0],
                    'emotions': json.loads(row[1]),
                    'response': json.loads(row[2]),
                    'model_used': row[3],
                    'timestamp': row[4]
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting user history: {e}")
            return []

class AuthManager:
    """Manage user authentication and registration"""
    
    def __init__(self):
        self.db_path = 'mental_health_users.db'
        self._init_database()
        logger.info("Auth manager initialized")
    
    def _init_database(self):
        """Initialize user authentication database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                name TEXT,
                age INTEGER,
                gender TEXT,
                country TEXT,
                mental_health_goals TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_login DATETIME
            )
        ''')
        
        # Create sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                token TEXT UNIQUE NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                expires_at DATETIME,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _generate_token(self) -> str:
        """Generate secure random token"""
        return secrets.token_urlsafe(32)
    
    def register_user(self, email: str, password: str, name: str, age: int, 
                     gender: str, country: str = None, mental_health_goals: str = None) -> Dict:
        """Register a new user"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if email already exists
            cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
            if cursor.fetchone():
                conn.close()
                return {'success': False, 'message': 'Email already registered'}
            
            # Hash password
            password_hash = self._hash_password(password)
            
            # Insert user
            cursor.execute('''
                INSERT INTO users (email, password_hash, name, age, gender, country, mental_health_goals)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (email, password_hash, name, age, gender, country, mental_health_goals))
            
            user_id = cursor.lastrowid
            
            # Generate session token
            token = self._generate_token()
            expires_at = datetime.now() + timedelta(days=30)
            
            cursor.execute('''
                INSERT INTO sessions (user_id, token, expires_at)
                VALUES (?, ?, ?)
            ''', (user_id, token, expires_at))
            
            conn.commit()
            conn.close()
            
            logger.info(f"New user registered: {email}")
            
            return {
                'success': True,
                'user_id': user_id,
                'token': token,
                'message': 'Registration successful'
            }
            
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return {'success': False, 'message': 'Registration failed'}
    
    def login_user(self, email: str, password: str) -> Dict:
        """Login user and create session"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get user
            password_hash = self._hash_password(password)
            cursor.execute('''
                SELECT id, name FROM users 
                WHERE email = ? AND password_hash = ?
            ''', (email, password_hash))
            
            user = cursor.fetchone()
            
            if not user:
                conn.close()
                return {'success': False, 'message': 'Invalid email or password'}
            
            user_id, name = user
            
            # Update last login
            cursor.execute('''
                UPDATE users SET last_login = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (user_id,))
            
            # Generate new session token
            token = self._generate_token()
            expires_at = datetime.now() + timedelta(days=30)
            
            cursor.execute('''
                INSERT INTO sessions (user_id, token, expires_at)
                VALUES (?, ?, ?)
            ''', (user_id, token, expires_at))
            
            conn.commit()
            conn.close()
            
            logger.info(f"User logged in: {email}")
            
            return {
                'success': True,
                'user_id': user_id,
                'name': name,
                'token': token,
                'message': 'Login successful'
            }
            
        except Exception as e:
            logger.error(f"Login error: {e}")
            return {'success': False, 'message': 'Login failed'}
    
    def verify_token(self, token: str) -> Dict:
        """Verify session token"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT s.user_id, u.email, u.name
                FROM sessions s
                JOIN users u ON s.user_id = u.id
                WHERE s.token = ? AND s.expires_at > CURRENT_TIMESTAMP
            ''', (token,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                user_id, email, name = result
                return {
                    'valid': True,
                    'user_id': user_id,
                    'email': email,
                    'name': name
                }
            else:
                return {'valid': False}
                
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return {'valid': False}
    
    def logout_user(self, token: str) -> bool:
        """Logout user by invalidating token"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM sessions WHERE token = ?', (token,))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Logout error: {e}")
            return False

# Initialize components
auth_manager = AuthManager()
hf_client = HuggingFaceModelClient()
empathy_engine = EmpathyEngine()
db_manager = DatabaseManager()
audio_generator = AudioGenerator()

# Utility functions
def log_request(f):
    """Decorator to log API requests"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        try:
            result = f(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"Request to {request.endpoint} completed in {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Request to {request.endpoint} failed after {duration:.3f}s: {e}")
            raise
    return decorated_function

def validate_text_input(text: str, max_length: int = 1000) -> str:
    """Validate and clean text input"""
    
    if not text or not isinstance(text, str):
        raise BadRequest("Text must be a non-empty string")
    
    text = text.strip()
    if len(text) == 0:
        raise BadRequest("Text cannot be empty")
    
    if len(text) > max_length:
        raise BadRequest(f"Text must be less than {max_length} characters")
    
    return text

# Error handlers
@app.errorhandler(400)
def bad_request(error):
    logger.warning(f"Bad request: {error}")
    return jsonify({
        'success': False,
        'error': 'Bad Request',
        'message': str(error),
        'timestamp': datetime.now().isoformat()
    }), 400

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal Server Error',
        'message': 'An unexpected error occurred',
        'timestamp': datetime.now().isoformat()
    }), 500

# API Routes
@app.route('/')
def index():
    """API documentation page"""
    return jsonify({
        'name': 'Mental Health Companion API',
        'version': '1.0.0',
        'description': 'Advanced emotion detection and empathetic response API',
        'endpoints': {
            'health': 'GET /api/health',
            'emotions': 'POST /api/emotions/detect',
            'chat': 'POST /api/chat/message',
            'models': 'GET /api/models/info',
            'history': 'GET /api/chat/history/<user_id>'
        },
        'models': {
            'logistic_regression': 'Classical ML - High recall',
            'enhanced_distilbert': 'Transformer - High precision'
        }
    })

# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

@app.route('/api/auth/register', methods=['POST'])
@limiter.limit("5 per hour")
def register():
    """
    Register a new user
    
    Expected JSON:
    {
        "email": "user@example.com",
        "password": "securepassword",
        "name": "John Doe",
        "age": 25,
        "gender": "male",
        "country": "USA",
        "mental_health_goals": "anxiety"
    }
    """
    try:
        if not request.is_json:
            raise BadRequest("Request must be JSON")
        
        data = request.get_json()
        
        # Validate required fields
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        name = data.get('name', '').strip()
        age = data.get('age')
        gender = data.get('gender', '').strip()
        country = data.get('country', '').strip()
        mental_health_goals = data.get('mental_health_goals', '').strip()
        
        if not email or not password or not name or not age or not gender:
            raise BadRequest("Missing required fields")
        
        if len(password) < 8:
            raise BadRequest("Password must be at least 8 characters")
        
        if not isinstance(age, int) or age < 13 or age > 120:
            raise BadRequest("Invalid age")
        
        # Register user
        result = auth_manager.register_user(
            email, password, name, age, gender, country, mental_health_goals
        )
        
        if result['success']:
            return jsonify(result), 201
        else:
            return jsonify(result), 400
            
    except BadRequest as e:
        return jsonify({'success': False, 'message': str(e)}), 400
    except Exception as e:
        logger.error(f"Registration endpoint error: {e}")
        return jsonify({'success': False, 'message': 'Registration failed'}), 500

@app.route('/api/auth/login', methods=['POST'])
@limiter.limit("10 per minute")
def login():
    """
    Login user
    
    Expected JSON:
    {
        "email": "user@example.com",
        "password": "securepassword"
    }
    """
    try:
        if not request.is_json:
            raise BadRequest("Request must be JSON")
        
        data = request.get_json()
        
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        if not email or not password:
            raise BadRequest("Email and password required")
        
        # Login user
        result = auth_manager.login_user(email, password)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 401
            
    except BadRequest as e:
        return jsonify({'success': False, 'message': str(e)}), 400
    except Exception as e:
        logger.error(f"Login endpoint error: {e}")
        return jsonify({'success': False, 'message': 'Login failed'}), 500

@app.route('/api/auth/verify', methods=['POST'])
def verify_token():
    """
    Verify authentication token
    
    Expected JSON:
    {
        "token": "user_token_here"
    }
    """
    try:
        if not request.is_json:
            raise BadRequest("Request must be JSON")
        
        data = request.get_json()
        token = data.get('token', '')
        
        if not token:
            raise BadRequest("Token required")
        
        result = auth_manager.verify_token(token)
        
        return jsonify(result), 200
            
    except Exception as e:
        logger.error(f"Token verification endpoint error: {e}")
        return jsonify({'valid': False}), 500

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    """
    Logout user
    
    Expected JSON:
    {
        "token": "user_token_here"
    }
    """
    try:
        if not request.is_json:
            raise BadRequest("Request must be JSON")
        
        data = request.get_json()
        token = data.get('token', '')
        
        if not token:
            raise BadRequest("Token required")
        
        success = auth_manager.logout_user(token)
        
        return jsonify({'success': success}), 200
            
    except Exception as e:
        logger.error(f"Logout endpoint error: {e}")
        return jsonify({'success': False}), 500

# ============================================================================
# HEALTH & INFO ENDPOINTS
# ============================================================================

@app.route('/api/gemini/status')
def gemini_status():
    """Check Gemini AI status"""
    try:
        status = {
            'gemini_library_installed': GEMINI_AVAILABLE,
            'gemini_api_key_set': bool(os.environ.get('GEMINI_API_KEY')),
            'gemini_initialized': empathy_engine.gemini_available,
            'message': ''
        }
        
        if empathy_engine.gemini_available:
            status['message'] = '✅ Gemini AI is active and ready'
        elif not GEMINI_AVAILABLE:
            status['message'] = '❌ google-generativeai library not installed'
        elif not os.environ.get('GEMINI_API_KEY'):
            status['message'] = '❌ GEMINI_API_KEY not set in environment'
        else:
            status['message'] = '❌ Gemini initialization failed'
        
        return jsonify(status), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models': {
            'huggingface_available': True,
            'empathy_engine': True,
            'database': True,
            'gemini_ai': empathy_engine.gemini_available,
            'audio_generation': audio_generator.audio_available
        }
    })

@app.route('/api/models/info')
def get_models_info():
    """Get information about available models"""
    return jsonify({
        'success': True,
        'models': {
            'logistic_regression': {
                'name': 'Mental Health Logistic Regression',
                'type': 'Classical ML',
                'performance': {
                    'f1_score': 0.298,
                    'precision': 0.219,
                    'recall': 0.508,
                    'accuracy': 0.901
                },
                'characteristics': 'High recall - good for detecting subtle emotions',
                'best_for': 'Comprehensive emotion detection'
            },
            'enhanced_distilbert': {
                'name': 'Enhanced Fine-tuned DistilBERT',
                'type': 'Transformer',
                'performance': {
                    'f1_score': 0.298,
                    'precision': 0.459,
                    'recall': 0.260,
                    'accuracy': 0.895
                },
                'characteristics': 'High precision - conservative predictions',
                'best_for': 'Accurate emotion classification'
            }
        },
        'emotions': [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
            'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]
    })

@app.route('/api/emotions/detect', methods=['POST'])
@limiter.limit("30 per minute")
@log_request
def detect_emotions():
    """
    Detect emotions from text
    
    Expected JSON:
    {
        "text": "I am feeling great today!",
        "model": "enhanced_distilbert" or "logistic_regression"
    }
    """
    try:
        if not request.is_json:
            raise BadRequest("Request must be JSON")
        
        data = request.get_json()
        
        # Validate input
        text = validate_text_input(data.get('text', ''))
        model_type = data.get('model', 'enhanced_distilbert')
        
        if model_type not in ['logistic_regression', 'enhanced_distilbert']:
            raise BadRequest("Model must be 'logistic_regression' or 'enhanced_distilbert'")
        
        # Detect emotions
        start_time = time.time()
        emotions = hf_client.predict_emotions(text, model_type)
        prediction_time = time.time() - start_time
        
        # Format response
        response = {
            'success': True,
            'input_text': text,
            'model_used': model_type,
            'prediction_time': round(prediction_time, 3),
            'emotions': emotions,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Emotion detection successful: {model_type} - {len(text)} chars")
        return jsonify(response)
        
    except BadRequest as e:
        return jsonify({
            'success': False,
            'error': 'Bad Request',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 400
        
    except Exception as e:
        logger.error(f"Emotion detection error: {e}\n{traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': 'Internal Server Error',
            'message': 'An error occurred during emotion detection',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/chat/message', methods=['POST'])
@limiter.limit("20 per minute")
@log_request
def chat_message():
    """
    Complete chat interaction with emotion detection and empathetic response
    
    Expected JSON:
    {
        "message": "I had a really tough day at work",
        "user_id": "user123",
        "model": "enhanced_distilbert",
        "use_advanced_ai": true
    }
    """
    try:
        if not request.is_json:
            raise BadRequest("Request must be JSON")
        
        data = request.get_json()
        
        # Validate input
        message = validate_text_input(data.get('message', ''))
        user_id = data.get('user_id', 'anonymous')
        model_type = data.get('model', 'enhanced_distilbert')
        use_advanced_ai = data.get('use_advanced_ai', True)
        
        if model_type not in ['logistic_regression', 'enhanced_distilbert']:
            raise BadRequest("Model must be 'logistic_regression' or 'enhanced_distilbert'")
        
        # Detect emotions
        start_time = time.time()
        emotions = hf_client.predict_emotions(message, model_type)
        emotion_time = time.time() - start_time
        
        # Generate empathetic response (force Gemini if requested)
        response_start = time.time()
        
        logger.info(f"🔍 Response generation - use_advanced_ai: {use_advanced_ai}, gemini_available: {empathy_engine.gemini_available}")
        
        if use_advanced_ai and empathy_engine.gemini_available:
            logger.info("🤖 Using Gemini AI for response generation")
            empathetic_response = empathy_engine._generate_advanced_response(message, emotions, emotions.get('top_emotions', [{}])[0].get('emotion', 'neutral'))
        else:
            if not empathy_engine.gemini_available:
                logger.warning("⚠️ Gemini not available, using template responses")
            empathetic_response = empathy_engine.generate_response(message, emotions)
        
        logger.info(f"📝 Response type: {empathetic_response.get('type', 'unknown')}")
        response_time = time.time() - response_start
        
        # Generate audio for the response (non-blocking)
        audio_start = time.time()
        audio_filename = None
        audio_error = None
        
        try:
            if empathetic_response.get('success') and empathetic_response.get('response'):
                response_text = empathetic_response['response']
                audio_filename = audio_generator.generate_audio(response_text)
                
                # Clean up old audio files periodically (only if audio works)
                if audio_filename:
                    audio_generator.cleanup_old_audio()
        except Exception as e:
            audio_error = str(e)
            logger.warning(f"Audio generation failed but continuing: {e}")
        
        audio_time = time.time() - audio_start
        
        # Save conversation
        db_manager.save_conversation(user_id, message, emotions, empathetic_response, model_type)
        
        # Format complete response
        complete_response = {
            'success': True,
            'user_message': message,
            'user_id': user_id,
            'emotion_analysis': emotions,
            'empathetic_response': empathetic_response,
            'audio': {
                'available': audio_filename is not None,
                'filename': audio_filename,
                'url': f'/api/audio/{audio_filename}' if audio_filename else None
            },
            'model_used': model_type,
            'processing_time': {
                'emotion_detection': round(emotion_time, 3),
                'response_generation': round(response_time, 3),
                'audio_generation': round(audio_time, 3),
                'total': round(emotion_time + response_time + audio_time, 3)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Chat interaction successful: {user_id} - {model_type}")
        return jsonify(complete_response)
        
    except BadRequest as e:
        return jsonify({
            'success': False,
            'error': 'Bad Request',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 400
        
    except Exception as e:
        logger.error(f"Chat interaction error: {e}\n{traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': 'Internal Server Error',
            'message': 'An error occurred during chat interaction',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/audio/<filename>')
def serve_audio(filename):
    """Serve generated audio files"""
    try:
        # Security: only allow audio files
        if not filename.endswith('.wav') or '..' in filename:
            return jsonify({'error': 'Invalid filename'}), 400
        
        filepath = os.path.join(audio_generator.audio_dir, filename)
        
        if os.path.exists(filepath):
            from flask import send_file
            return send_file(filepath, mimetype='audio/wav')
        else:
            return jsonify({'error': 'Audio file not found'}), 404
            
    except Exception as e:
        logger.error(f"Error serving audio file: {e}")
        return jsonify({'error': 'Failed to serve audio'}), 500

@app.route('/api/chat/history/<user_id>')
def get_chat_history(user_id):
    """Get user's conversation history"""
    try:
        limit = request.args.get('limit', 10, type=int)
        limit = min(limit, 50)  # Max 50 conversations
        
        history = db_manager.get_user_history(user_id, limit)
        
        return jsonify({
            'success': True,
            'user_id': user_id,
            'conversations': history,
            'count': len(history),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal Server Error',
            'message': 'Failed to retrieve chat history',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/models/compare', methods=['POST'])
@limiter.limit("10 per minute")
@log_request
def compare_models():
    """
    Compare both models on the same input
    
    Expected JSON:
    {
        "text": "I am feeling anxious about tomorrow"
    }
    """
    try:
        if not request.is_json:
            raise BadRequest("Request must be JSON")
        
        data = request.get_json()
        text = validate_text_input(data.get('text', ''))
        
        # Test both models
        start_time = time.time()
        
        lr_result = hf_client.predict_emotions(text, 'logistic_regression')
        bert_result = hf_client.predict_emotions(text, 'enhanced_distilbert')
        
        total_time = time.time() - start_time
        
        # Compare results
        comparison = {
            'success': True,
            'input_text': text,
            'models': {
                'logistic_regression': lr_result,
                'enhanced_distilbert': bert_result
            },
            'comparison': {
                'lr_emotions': len(lr_result.get('predicted_emotions', [])),
                'bert_emotions': len(bert_result.get('predicted_emotions', [])),
                'lr_confidence': lr_result.get('confidence_score', 0.0),
                'bert_confidence': bert_result.get('confidence_score', 0.0),
                'agreement': _calculate_emotion_agreement(lr_result, bert_result)
            },
            'processing_time': round(total_time, 3),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(comparison)
        
    except Exception as e:
        logger.error(f"Model comparison error: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal Server Error',
            'message': 'Failed to compare models',
            'timestamp': datetime.now().isoformat()
        }), 500

def _calculate_emotion_agreement(lr_result: Dict, bert_result: Dict) -> Dict:
    """Calculate agreement between model predictions"""
    
    try:
        lr_emotions = set([e['emotion'] for e in lr_result.get('predicted_emotions', [])])
        bert_emotions = set([e['emotion'] for e in bert_result.get('predicted_emotions', [])])
        
        common = lr_emotions.intersection(bert_emotions)
        lr_only = lr_emotions - bert_emotions
        bert_only = bert_emotions - lr_emotions
        
        return {
            'common_emotions': list(common),
            'lr_only': list(lr_only),
            'bert_only': list(bert_only),
            'agreement_score': len(common) / max(len(lr_emotions.union(bert_emotions)), 1)
        }
        
    except Exception:
        return {'agreement_score': 0.0, 'common_emotions': [], 'lr_only': [], 'bert_only': []}

if __name__ == '__main__':
    # Production configuration
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting Mental Health Companion API on port {port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"HuggingFace models: {list(hf_client.models.keys())}")
    logger.info(f"Gemini AI available: {empathy_engine.gemini_available}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )