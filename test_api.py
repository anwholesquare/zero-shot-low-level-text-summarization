#!/usr/bin/env python3
"""
Test script for AI Services Flask API
This script demonstrates how to use the API endpoints
"""

import requests
import json
import time
from typing import Dict, Any

class AIServicesAPITester:
    """Test client for AI Services API"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })
    
    def test_health_check(self) -> Dict[str, Any]:
        """Test the health check endpoint"""
        print("🔍 Testing health check endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/")
            response.raise_for_status()
            result = response.json()
            print("✅ Health check passed")
            print(f"📊 Available services: {result.get('available_services', [])}")
            return result
        except Exception as e:
            print(f"❌ Health check failed: {str(e)}")
            return {"error": str(e)}
    
    def test_openai_chat(self, prompt: str = "Hello, how are you?") -> Dict[str, Any]:
        """Test OpenAI chat endpoint"""
        print(f"\n🤖 Testing OpenAI chat with prompt: '{prompt}'")
        try:
            payload = {
                "prompt": prompt,
                "model": "gpt-3.5-turbo",
                "max_tokens": 100,
                "temperature": 0.7
            }
            
            response = self.session.post(
                f"{self.base_url}/api/openai/chat",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            print("✅ OpenAI chat successful")
            print(f"📝 Response: {result.get('response', 'No response')[:100]}...")
            return result
        except Exception as e:
            print(f"❌ OpenAI chat failed: {str(e)}")
            return {"error": str(e)}
    
    def test_anthropic_chat(self, prompt: str = "Explain machine learning briefly") -> Dict[str, Any]:
        """Test Anthropic chat endpoint"""
        print(f"\n🧠 Testing Anthropic chat with prompt: '{prompt}'")
        try:
            payload = {
                "prompt": prompt,
                "model": "claude-3-5-haiku-20241022",
                "max_tokens": 100,
                "temperature": 0.7
            }
            
            response = self.session.post(
                f"{self.base_url}/api/anthropic/chat",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            print("✅ Anthropic chat successful")
            print(f"📝 Response: {result.get('response', 'No response')[:100]}...")
            return result
        except Exception as e:
            print(f"❌ Anthropic chat failed: {str(e)}")
            return {"error": str(e)}
    
    def test_deepseek_chat(self, prompt: str = "Write a simple Python function") -> Dict[str, Any]:
        """Test DeepSeek chat endpoint"""
        print(f"\n💻 Testing DeepSeek chat with prompt: '{prompt}'")
        try:
            payload = {
                "prompt": prompt,
                "model": "deepseek-chat",
                "max_tokens": 100,
                "temperature": 0.7
            }
            
            response = self.session.post(
                f"{self.base_url}/api/deepseek/chat",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            print("✅ DeepSeek chat successful")
            print(f"📝 Response: {result.get('response', 'No response')[:100]}...")
            return result
        except Exception as e:
            print(f"❌ DeepSeek chat failed: {str(e)}")
            return {"error": str(e)}
    
    def test_gemini_chat(self, prompt: str = "What is artificial intelligence?") -> Dict[str, Any]:
        """Test Gemini chat endpoint"""
        print(f"\n🌟 Testing Gemini chat with prompt: '{prompt}'")
        try:
            payload = {
                "prompt": prompt,
                "model": "gemini-2.0-flash-lite",
                "max_tokens": 100,
                "temperature": 0.7
            }
            
            response = self.session.post(
                f"{self.base_url}/api/gemini/chat",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            print("✅ Gemini chat successful")
            print(f"📝 Response: {result.get('response', 'No response')[:100]}...")
            return result
        except Exception as e:
            print(f"❌ Gemini chat failed: {str(e)}")
            return {"error": str(e)}
    
    def test_bloomz_generate(self, prompt: str = "The future of AI is") -> Dict[str, Any]:
        """Test Bloomz text generation endpoint"""
        print(f"\n🌸 Testing Bloomz generation with prompt: '{prompt}'")
        try:
            payload = {
                "prompt": prompt,
                "max_length": 50,
                "temperature": 0.7
            }
            
            response = self.session.post(
                f"{self.base_url}/api/bloomz/generate",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            print("✅ Bloomz generation successful")
            print(f"📝 Response: {result.get('response', 'No response')[:100]}...")
            return result
        except Exception as e:
            print(f"❌ Bloomz generation failed: {str(e)}")
            return {"error": str(e)}
    
    def test_compare_services(self, prompt: str = "What is Python?") -> Dict[str, Any]:
        """Test multi-service comparison endpoint"""
        print(f"\n🔄 Testing service comparison with prompt: '{prompt}'")
        try:
            payload = {
                "prompt": prompt,
                "services": ["openai", "anthropic", "gemini"]
            }
            
            response = self.session.post(
                f"{self.base_url}/api/compare",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            print("✅ Service comparison successful")
            
            results = result.get('results', {})
            for service, response_text in results.items():
                print(f"📝 {service.upper()}: {str(response_text)[:80]}...")
            
            return result
        except Exception as e:
            print(f"❌ Service comparison failed: {str(e)}")
            return {"error": str(e)}
    
    def run_all_tests(self):
        """Run all API tests"""
        print("🚀 Starting AI Services API Tests")
        print("=" * 50)
        
        # Test health check first
        health_result = self.test_health_check()
        if "error" in health_result:
            print("❌ API is not running. Please start the Flask app first.")
            return
        
        # Wait a moment between tests
        time.sleep(1)
        
        # Test individual services
        test_methods = [
            self.test_openai_chat,
            self.test_anthropic_chat,
            self.test_deepseek_chat,
            self.test_gemini_chat,
            self.test_bloomz_generate,
            self.test_compare_services
        ]
        
        results = {}
        for test_method in test_methods:
            try:
                result = test_method()
                results[test_method.__name__] = result
                time.sleep(1)  # Rate limiting
            except Exception as e:
                print(f"❌ Test {test_method.__name__} failed: {str(e)}")
                results[test_method.__name__] = {"error": str(e)}
        
        print("\n" + "=" * 50)
        print("📊 Test Summary:")
        
        successful_tests = 0
        total_tests = len(test_methods)
        
        for test_name, result in results.items():
            status = "✅ PASSED" if "error" not in result else "❌ FAILED"
            if "error" not in result:
                successful_tests += 1
            print(f"{test_name}: {status}")
        
        print(f"\n🎯 Success Rate: {successful_tests}/{total_tests} ({(successful_tests/total_tests)*100:.1f}%)")
        
        if successful_tests == total_tests:
            print("🎉 All tests passed! Your AI Services API is working perfectly.")
        else:
            print("⚠️  Some tests failed. Check your API keys and service configurations.")

def main():
    """Main function to run the tests"""
    print("AI Services API Tester")
    print("Make sure your Flask app is running on http://localhost:5000")
    print()
    
    # Check if user wants to proceed
    try:
        input("Press Enter to start testing (Ctrl+C to cancel)...")
    except KeyboardInterrupt:
        print("\n👋 Testing cancelled.")
        return
    
    # Create tester instance and run tests
    tester = AIServicesAPITester()
    tester.run_all_tests()

if __name__ == "__main__":
    main() 