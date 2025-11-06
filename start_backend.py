#!/usr/bin/env python3
"""
Start the Mental Health Companion Backend
Simple startup script with environment setup
"""

import os
import sys
from dotenv import load_dotenv

def setup_environment():
    """Setup environment variables"""
    
    # Load .env file if it exists
    if os.path.exists('.env'):
        load_dotenv('.env')
        print("✅ Loaded environment from .env file")
    else:
        print("⚠️ No .env file found, using system environment")
    
    # Set default values
    os.environ.setdefault('FLASK_ENV', 'development')
    os.environ.setdefault('PORT', '5000')
    os.environ.setdefault('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    print(f"🔧 Environment: {os.environ.get('FLASK_ENV')}")
    print(f"🌐 Port: {os.environ.get('PORT')}")
    
    # Check optional configurations
    if os.environ.get('HUGGINGFACE_TOKEN'):
        print("✅ HuggingFace token configured")
    else:
        print("⚠️ HuggingFace token not set (using public models)")
    
    if os.environ.get('GEMINI_API_KEY'):
        print("✅ Gemini AI configured")
    else:
        print("⚠️ Gemini AI not configured (using template responses)")

def check_dependencies():
    """Check if required packages are installed"""
    
    required_packages = [
        'flask',
        'flask_cors',
        'flask_limiter',
        'requests',
        'google.generativeai'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print(f"💡 Install with: pip install -r requirements.txt")
        return False
    else:
        print("✅ All required packages installed")
        return True

def main():
    """Main startup function"""
    
    print("🚀 Mental Health Companion Backend Startup")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Cannot start backend due to missing dependencies")
        sys.exit(1)
    
    print("\n🏥 Starting Flask backend...")
    print("📊 Available endpoints:")
    print("   GET  /api/health - Health check")
    print("   GET  /api/models/info - Model information")
    print("   POST /api/emotions/detect - Emotion detection")
    print("   POST /api/chat/message - Complete chat interaction")
    print("   POST /api/models/compare - Compare both models")
    print("   GET  /api/chat/history/<user_id> - Chat history")
    
    print(f"\n🌐 Backend will be available at: http://localhost:{os.environ.get('PORT')}")
    print("🔄 Starting Flask application...")
    print("=" * 50)
    
    # Import and run the Flask app
    try:
        from app import app
        
        port = int(os.environ.get('PORT', 5000))
        debug = os.environ.get('FLASK_ENV') == 'development'
        
        app.run(
            host='0.0.0.0',
            port=port,
            debug=debug,
            threaded=True
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 Backend stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting backend: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()