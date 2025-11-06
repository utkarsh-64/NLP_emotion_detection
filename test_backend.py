#!/usr/bin/env python3
"""
Test the Flask Backend Locally
Verify all endpoints work correctly
"""

import requests
import json
import time

class BackendTester:
    """Test the Flask backend"""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.test_results = {}
    
    def test_health_check(self):
        """Test health check endpoint"""
        print("🏥 Testing health check...")
        
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Health check passed")
                print(f"   📊 Status: {data.get('status')}")
                print(f"   🤖 Models available: {data.get('models', {})}")
                return True
            else:
                print(f"   ❌ Health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"   ❌ Health check error: {e}")
            return False
    
    def test_models_info(self):
        """Test models info endpoint"""
        print("\n🤖 Testing models info...")
        
        try:
            response = requests.get(f"{self.base_url}/api/models/info", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', {})
                print(f"   ✅ Models info retrieved")
                print(f"   📊 Available models: {list(models.keys())}")
                
                for model_name, model_info in models.items():
                    f1_score = model_info.get('performance', {}).get('f1_score', 0)
                    print(f"      {model_name}: F1={f1_score}")
                
                return True
            else:
                print(f"   ❌ Models info failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"   ❌ Models info error: {e}")
            return False
    
    def test_emotion_detection(self, model_type="enhanced_distilbert"):
        """Test emotion detection endpoint"""
        print(f"\n🧠 Testing emotion detection ({model_type})...")
        
        test_cases = [
            "I am very happy today!",
            "I feel sad and disappointed",
            "This is really confusing",
            "I'm excited about tomorrow"
        ]
        
        successful_tests = 0
        
        for i, text in enumerate(test_cases, 1):
            try:
                payload = {
                    "text": text,
                    "model": model_type
                }
                
                response = requests.post(
                    f"{self.base_url}/api/emotions/detect",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    emotions = data.get('emotions', {})
                    top_emotions = emotions.get('top_emotions', [])
                    
                    print(f"   ✅ Test {i}: '{text[:30]}...'")
                    if top_emotions:
                        top_emotion = top_emotions[0]
                        print(f"      Top emotion: {top_emotion.get('emotion')} ({top_emotion.get('confidence', 0):.3f})")
                    
                    successful_tests += 1
                else:
                    print(f"   ❌ Test {i} failed: {response.status_code}")
                    print(f"      Response: {response.text[:100]}")
                    
            except Exception as e:
                print(f"   ❌ Test {i} error: {e}")
        
        success_rate = (successful_tests / len(test_cases)) * 100
        print(f"   📊 Success rate: {successful_tests}/{len(test_cases)} ({success_rate:.1f}%)")
        
        return successful_tests > 0
    
    def test_chat_interaction(self):
        """Test complete chat interaction"""
        print(f"\n💬 Testing chat interaction...")
        
        try:
            payload = {
                "message": "I had a really tough day at work and I'm feeling overwhelmed",
                "user_id": "test_user_123",
                "model": "enhanced_distilbert",
                "use_advanced_ai": True
            }
            
            response = requests.post(
                f"{self.base_url}/api/chat/message",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"   ✅ Chat interaction successful")
                
                # Show emotion analysis
                emotions = data.get('emotion_analysis', {})
                predicted_emotions = emotions.get('predicted_emotions', [])
                if predicted_emotions:
                    print(f"   🎯 Detected emotions: {[e['emotion'] for e in predicted_emotions[:3]]}")
                
                # Show empathetic response
                empathetic_response = data.get('empathetic_response', {})
                response_text = empathetic_response.get('response', '')
                if response_text:
                    print(f"   💝 Response: {response_text[:100]}...")
                
                # Show processing time
                processing_time = data.get('processing_time', {})
                total_time = processing_time.get('total', 0)
                print(f"   ⏱️ Processing time: {total_time:.3f}s")
                
                return True
            else:
                print(f"   ❌ Chat interaction failed: {response.status_code}")
                print(f"      Response: {response.text[:200]}")
                return False
                
        except Exception as e:
            print(f"   ❌ Chat interaction error: {e}")
            return False
    
    def test_model_comparison(self):
        """Test model comparison endpoint"""
        print(f"\n🔬 Testing model comparison...")
        
        try:
            payload = {
                "text": "I'm feeling anxious about my presentation tomorrow"
            }
            
            response = requests.post(
                f"{self.base_url}/api/models/compare",
                json=payload,
                timeout=45
            )
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"   ✅ Model comparison successful")
                
                # Show comparison results
                comparison = data.get('comparison', {})
                lr_emotions = comparison.get('lr_emotions', 0)
                bert_emotions = comparison.get('bert_emotions', 0)
                agreement = comparison.get('agreement_score', 0)
                
                print(f"   📊 Logistic Regression: {lr_emotions} emotions")
                print(f"   📊 Enhanced DistilBERT: {bert_emotions} emotions")
                print(f"   🤝 Agreement score: {agreement:.3f}")
                
                return True
            else:
                print(f"   ❌ Model comparison failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"   ❌ Model comparison error: {e}")
            return False
    
    def run_all_tests(self):
        """Run all backend tests"""
        
        print("🧪 FLASK BACKEND TESTING")
        print(f"🌐 Base URL: {self.base_url}")
        print("=" * 60)
        
        tests = [
            ("Health Check", self.test_health_check),
            ("Models Info", self.test_models_info),
            ("Emotion Detection (DistilBERT)", lambda: self.test_emotion_detection("enhanced_distilbert")),
            ("Emotion Detection (Logistic)", lambda: self.test_emotion_detection("logistic_regression")),
            ("Chat Interaction", self.test_chat_interaction),
            ("Model Comparison", self.test_model_comparison)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            try:
                results[test_name] = test_func()
            except Exception as e:
                print(f"   ❌ {test_name} crashed: {e}")
                results[test_name] = False
        
        # Summary
        print(f"\n🏆 TEST SUMMARY")
        print("=" * 40)
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, passed_test in results.items():
            status = "✅ PASSED" if passed_test else "❌ FAILED"
            print(f"   {test_name}: {status}")
        
        print(f"\n📊 Overall: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
        
        if passed == total:
            print(f"🎉 ALL TESTS PASSED! Backend is ready for deployment!")
        elif passed > total // 2:
            print(f"⚠️ Most tests passed. Check failed tests and HuggingFace model status.")
        else:
            print(f"❌ Many tests failed. Check backend setup and model availability.")
        
        return passed == total

def main():
    """Main testing function"""
    
    print("🚀 Mental Health Companion Backend Tester")
    print("💡 Make sure to start the Flask backend first:")
    print("   python app.py")
    print()
    
    # Wait a moment for user to start backend
    input("Press Enter when backend is running on localhost:5000...")
    
    tester = BackendTester()
    success = tester.run_all_tests()
    
    if success:
        print(f"\n🚀 NEXT STEPS:")
        print("1. ✅ Backend is working locally")
        print("2. 🌐 Deploy to Render")
        print("3. 📱 Build React Native frontend")
        print("4. 🔗 Connect mobile app to deployed backend")
    else:
        print(f"\n🔧 TROUBLESHOOTING:")
        print("1. Check if Flask backend is running")
        print("2. Verify HuggingFace models are processed")
        print("3. Check environment variables")
        print("4. Review backend logs for errors")

if __name__ == "__main__":
    main()