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
from datetime import datetime
from functools import wraps
import traceback
import sqlite3
from werkzeug.exceptions import BadRequest, InternalServerError
import google.generativeai as genai
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
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

class HuggingFaceModelClient:
    """Client for interacting with HuggingFace models"""
    
    def __init__(self):
        self.username = "Utkarsh64"
        self.models = {
            'logistic_regression': f"{self.username}/mental-health-logistic-regression",
            'enhanced_distilbert': f"{self.username}/huggingface_distilbert_model"
        }
        
        # Multiple API endpoints (HuggingFace is transitioning)
        self.api_endpoints = [
            "https://api-inference.huggingface.co/models",
            "https://router.huggingface.co/hf-inference/models"
        ]
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('HUGGINGFACE_TOKEN', '')}"
        }
        
        # Remove auth header if no token
        if not os.environ.get('HUGGINGFACE_TOKEN'):
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
        
        # Try multiple API endpoints
        for endpoint in self.api_endpoints:
            api_url = f"{endpoint}/{model_name}"
            
            try:
                logger.info(f"Trying HuggingFace API: {api_url}")
                
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
                    logger.warning(f"Model loading (503): {model_type}")
                    continue
                
                elif response.status_code == 401:
                    logger.warning(f"Authentication required (401): {api_url}")
                    continue
                
                elif response.status_code == 410:
                    logger.warning(f"API endpoint deprecated (410): {api_url}")
                    continue
                
                else:
                    logger.error(f"HuggingFace API error {response.status_code}: {response.text}")
                    continue
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout for {api_url}")
                continue
            except Exception as e:
                logger.error(f"Request error for {api_url}: {e}")
                continue
        
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
        
        # Simple keyword-based emotion detection as fallback
        emotion_keywords = {
            'joy': ['happy', 'excited', 'great', 'amazing', 'wonderful', 'fantastic', 'good'],
            'sadness': ['sad', 'depressed', 'down', 'upset', 'disappointed', 'hurt'],
            'anger': ['angry', 'mad', 'furious', 'annoyed', 'frustrated', 'irritated'],
            'fear': ['scared', 'afraid', 'anxious', 'worried', 'nervous', 'terrified'],
            'love': ['love', 'adore', 'cherish', 'care', 'affection'],
            'gratitude': ['thank', 'grateful', 'appreciate', 'thankful'],
            'confusion': ['confused', 'puzzled', 'unclear', 'don\'t understand'],
            'excitement': ['excited', 'thrilled', 'pumped', 'enthusiastic'],
            'neutral': []
        }
        
        text_lower = text.lower()
        detected_emotions = []
        
        for emotion, keywords in emotion_keywords.items():
            score = 0.0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 0.3
            
            if score > 0:
                detected_emotions.append({
                    'emotion': emotion,
                    'confidence': min(score, 0.9),
                    'predicted': score > 0.3
                })
        
        # Add neutral if no emotions detected
        if not detected_emotions:
            detected_emotions.append({
                'emotion': 'neutral',
                'confidence': 0.7,
                'predicted': True
            })
        
        # Sort by confidence
        detected_emotions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'success': True,
            'model_used': f"{model_type}_fallback",
            'emotions': detected_emotions,
            'top_emotions': detected_emotions[:5],
            'predicted_emotions': [e for e in detected_emotions if e['predicted']],
            'confidence_score': detected_emotions[0]['confidence'] if detected_emotions else 0.0,
            'source': 'fallback_keywords',
            'note': 'Using fallback detection - HuggingFace models are processing'
        }

class EmpathyEngine:
    """Generate empathetic responses based on detected emotions"""
    
    def __init__(self):
        # Initialize Gemini API if available
        self.gemini_available = False
        try:
            gemini_key = os.environ.get('GEMINI_API_KEY')
            if gemini_key:
                genai.configure(api_key=gemini_key)
                self.gemini_model = genai.GenerativeModel('gemini-pro')
                self.gemini_available = True
                logger.info("Gemini API initialized successfully")
        except Exception as e:
            logger.warning(f"Gemini API not available: {e}")
        
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
            if self._should_use_advanced_response(emotions, text):
                return self._generate_advanced_response(text, emotions, primary_emotion)
            else:
                return self._generate_template_response(text, emotions, primary_emotion)
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._generate_fallback_response()
    
    def _should_use_advanced_response(self, emotions: Dict[str, Any], text: str) -> bool:
        """Decide whether to use advanced AI response"""
        
        # Use advanced response for:
        # 1. Complex emotional states (multiple emotions)
        # 2. Low confidence predictions
        # 3. Longer, more complex text
        
        predicted_emotions = emotions.get('predicted_emotions', [])
        confidence = emotions.get('confidence_score', 0.0)
        
        return (
            self.gemini_available and
            (len(predicted_emotions) > 2 or confidence < 0.6 or len(text) > 100)
        )
    
    def _generate_advanced_response(self, text: str, emotions: Dict[str, Any], primary_emotion: str) -> Dict[str, Any]:
        """Generate response using advanced AI"""
        
        try:
            # Create context-rich prompt
            predicted_emotions = emotions.get('predicted_emotions', [])
            emotion_list = [e['emotion'] for e in predicted_emotions[:3]]
            
            prompt = f"""You are a compassionate mental health companion. A user has shared: "{text}"

My emotion detection system identified these emotions: {', '.join(emotion_list) if emotion_list else primary_emotion}
Primary emotion: {primary_emotion}
Confidence: {emotions.get('confidence_score', 0.0):.2f}

Provide a warm, empathetic response (100-150 words) that:
1. Acknowledges their emotional state
2. Validates their feelings
3. Offers gentle support
4. Asks a thoughtful follow-up question if appropriate

Be genuine, supportive, and professional. Avoid giving medical advice."""

            response = self.gemini_model.generate_content(prompt)
            
            return {
                'success': True,
                'response': response.text,
                'type': 'advanced_ai',
                'primary_emotion': primary_emotion,
                'emotions_addressed': emotion_list,
                'confidence': emotions.get('confidence_score', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Advanced AI response failed: {e}")
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

# Initialize components
hf_client = HuggingFaceModelClient()
empathy_engine = EmpathyEngine()
db_manager = DatabaseManager()

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
            'gemini_ai': empathy_engine.gemini_available
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
        
        # Generate empathetic response
        response_start = time.time()
        empathetic_response = empathy_engine.generate_response(message, emotions)
        response_time = time.time() - response_start
        
        # Save conversation
        db_manager.save_conversation(user_id, message, emotions, empathetic_response, model_type)
        
        # Format complete response
        complete_response = {
            'success': True,
            'user_message': message,
            'user_id': user_id,
            'emotion_analysis': emotions,
            'empathetic_response': empathetic_response,
            'model_used': model_type,
            'processing_time': {
                'emotion_detection': round(emotion_time, 3),
                'response_generation': round(response_time, 3),
                'total': round(emotion_time + response_time, 3)
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