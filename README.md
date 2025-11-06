# Mental Health Companion - Flask Backend

A production-ready Flask API for emotion detection and empathetic response generation, designed for mental health applications.

## 🎯 Features

- **Dual Model Support**: Choose between Logistic Regression (high recall) and Enhanced DistilBERT (high precision)
- **HuggingFace Integration**: Uses your trained models deployed on HuggingFace
- **Empathetic Responses**: AI-powered empathetic response generation
- **Conversation History**: SQLite database for user conversations
- **Rate Limiting**: Built-in protection against abuse
- **CORS Enabled**: Ready for React Native frontend
- **Production Ready**: Configured for Render deployment

## 🏗️ Architecture

```
React Native App
    ↓ HTTP/REST API
Flask Backend (Render)
    ↓ API Calls
HuggingFace Models
    ↓ Response Generation
Empathy Engine (Templates + Gemini AI)
```

## 📊 Your Models

### Logistic Regression
- **F1-Score**: 0.298
- **Precision**: 0.219
- **Recall**: 0.508
- **Best for**: Comprehensive emotion detection

### Enhanced DistilBERT
- **F1-Score**: 0.298
- **Precision**: 0.459
- **Recall**: 0.260
- **Best for**: Accurate emotion classification

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup Environment
```bash
cp .env.example .env
# Edit .env with your API keys (optional)
```

### 3. Start Backend
```bash
python start_backend.py
```

### 4. Test Backend
```bash
# In another terminal
python test_backend.py
```

## 🌐 API Endpoints

### Health Check
```http
GET /api/health
```

### Model Information
```http
GET /api/models/info
```

### Emotion Detection
```http
POST /api/emotions/detect
Content-Type: application/json

{
  "text": "I am feeling great today!",
  "model": "enhanced_distilbert"
}
```

### Complete Chat Interaction
```http
POST /api/chat/message
Content-Type: application/json

{
  "message": "I had a tough day at work",
  "user_id": "user123",
  "model": "enhanced_distilbert",
  "use_advanced_ai": true
}
```

### Model Comparison
```http
POST /api/models/compare
Content-Type: application/json

{
  "text": "I'm feeling anxious about tomorrow"
}
```

### Chat History
```http
GET /api/chat/history/user123?limit=10
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `FLASK_ENV` | Environment (development/production) | No |
| `PORT` | Server port (default: 5000) | No |
| `SECRET_KEY` | Flask secret key | Yes (production) |
| `HUGGINGFACE_TOKEN` | HuggingFace API token | No* |
| `GEMINI_API_KEY` | Gemini AI API key | No* |

*Optional but recommended for better performance

### HuggingFace Models

The backend automatically uses your deployed models:
- `Utkarsh64/mental-health-logistic-regression`
- `Utkarsh64/huggingface_distilbert_model`

## 🚀 Deployment to Render

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "Initial backend commit"
git push origin main
```

### 2. Deploy to Render
1. Go to [render.com](https://render.com)
2. Connect your GitHub repository
3. Choose "Web Service"
4. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT app:app`
   - **Environment**: Python 3

### 3. Set Environment Variables
In Render dashboard, add:
- `FLASK_ENV=production`
- `SECRET_KEY=<generate-random-key>`
- `HUGGINGFACE_TOKEN=<your-token>` (optional)
- `GEMINI_API_KEY=<your-key>` (optional)

## 📱 React Native Integration

Your backend is designed to work seamlessly with React Native:

```javascript
const API_BASE = 'https://your-app.render.com/api';

// Emotion detection
const detectEmotions = async (text, model = 'enhanced_distilbert') => {
  const response = await fetch(`${API_BASE}/emotions/detect`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, model })
  });
  return response.json();
};

// Complete chat interaction
const sendMessage = async (message, userId, model = 'enhanced_distilbert') => {
  const response = await fetch(`${API_BASE}/chat/message`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      message, 
      user_id: userId, 
      model,
      use_advanced_ai: true 
    })
  });
  return response.json();
};
```

## 🧪 Testing

### Local Testing
```bash
# Start backend
python start_backend.py

# Run tests (in another terminal)
python test_backend.py
```

### Manual Testing
1. Health check: `curl http://localhost:5000/api/health`
2. Emotion detection: Use the test script or Postman
3. Check logs: Monitor console output

## 📊 Performance

- **Response Time**: 200-2000ms (depending on HuggingFace API)
- **Rate Limits**: 30 requests/minute for emotions, 20 for chat
- **Fallback**: Keyword-based detection when HuggingFace is unavailable
- **Caching**: Built-in response caching for better performance

## 🔒 Security Features

- **Rate Limiting**: Prevents API abuse
- **Input Validation**: Sanitizes all user inputs
- **CORS Protection**: Configured for your frontend
- **Error Handling**: Comprehensive error responses
- **Logging**: Detailed request/response logging

## 🐛 Troubleshooting

### HuggingFace API Issues
- **410 Error**: API endpoints are transitioning (normal)
- **401 Error**: Add HuggingFace token to environment
- **503 Error**: Models are still processing (wait 10-15 minutes)

### Backend Issues
- **Import Errors**: Run `pip install -r requirements.txt`
- **Port Conflicts**: Change PORT in .env file
- **Database Issues**: Delete `mental_health_conversations.db` to reset

### Deployment Issues
- **Build Failures**: Check requirements.txt syntax
- **Runtime Errors**: Check Render logs for details
- **Environment Variables**: Ensure all required vars are set

## 📈 Monitoring

The backend includes comprehensive logging:
- Request/response times
- Error tracking
- Model performance
- User interaction patterns

Logs are available in:
- Local: `mental_health_api.log`
- Render: Dashboard logs section

## 🚀 Next Steps

1. ✅ Backend deployed and tested
2. 📱 Build React Native frontend
3. 🔗 Connect mobile app to backend
4. 🧪 End-to-end testing
5. 📊 Monitor performance and usage

## 🤝 Support

Your backend is production-ready with:
- Robust error handling
- Fallback mechanisms
- Comprehensive logging
- Rate limiting
- Security features

Ready for your React Native app! 🎉