#!/usr/bin/env python3
"""
Service Test Script for AI Services
This script tests each AI service individually to check if they're working properly
"""

import os
import sys
from dotenv import load_dotenv
from loguru import logger
import time

# Load environment variables
load_dotenv()

# Import AI service clients
try:
    from services.openai_service import OpenAIService
    from services.anthropic_service import AnthropicService
    from services.deepseek_service import DeepSeekService
    from services.bloomz_service import BloomzService
except ImportError as e:
    print(f"❌ Error importing services: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

class ServiceTester:
    """Test class for AI services"""
    
    def __init__(self):
        self.test_prompt = "Summarize this text: The quick brown fox jumps over the lazy dog. This is a simple test sentence."
        self.results = {}
    
    def test_openai_service(self):
        """Test OpenAI service"""
        print("\n🤖 Testing OpenAI Service...")
        
        try:
            service = OpenAIService()
            
            if not service.is_configured():
                print("❌ OpenAI: Not configured (missing API key)")
                self.results['openai'] = {'status': 'not_configured', 'error': 'Missing API key'}
                return
            
            print("   ✅ OpenAI: API key found")
            print("   🔄 Testing chat completion...")
            
            start_time = time.time()
            response = service.chat_completion(
                prompt=self.test_prompt,
                model="gpt-3.5-turbo",
                max_tokens=100,
                temperature=0.7
            )
            end_time = time.time()
            
            if response and len(response.strip()) > 0:
                print(f"   ✅ OpenAI: Chat completion successful ({end_time - start_time:.2f}s)")
                print(f"   📝 Response: {response[:100]}...")
                self.results['openai'] = {
                    'status': 'working',
                    'response_time': round(end_time - start_time, 2),
                    'response_length': len(response),
                    'sample_response': response[:100] + "..." if len(response) > 100 else response
                }
            else:
                print("   ❌ OpenAI: Empty response received")
                self.results['openai'] = {'status': 'error', 'error': 'Empty response'}
                
        except Exception as e:
            print(f"   ❌ OpenAI: Error - {str(e)}")
            self.results['openai'] = {'status': 'error', 'error': str(e)}
    
    def test_anthropic_service(self):
        """Test Anthropic service"""
        print("\n🧠 Testing Anthropic Service...")
        
        try:
            service = AnthropicService()
            
            if not service.is_configured():
                print("❌ Anthropic: Not configured (missing API key)")
                self.results['anthropic'] = {'status': 'not_configured', 'error': 'Missing API key'}
                return
            
            print("   ✅ Anthropic: API key found")
            print("   🔄 Testing chat completion...")
            
            start_time = time.time()
            response = service.chat_completion(
                prompt=self.test_prompt,
                model="claude-3-sonnet-20240229",
                max_tokens=100,
                temperature=0.7
            )
            end_time = time.time()
            
            if response and len(response.strip()) > 0:
                print(f"   ✅ Anthropic: Chat completion successful ({end_time - start_time:.2f}s)")
                print(f"   📝 Response: {response[:100]}...")
                self.results['anthropic'] = {
                    'status': 'working',
                    'response_time': round(end_time - start_time, 2),
                    'response_length': len(response),
                    'sample_response': response[:100] + "..." if len(response) > 100 else response
                }
            else:
                print("   ❌ Anthropic: Empty response received")
                self.results['anthropic'] = {'status': 'error', 'error': 'Empty response'}
                
        except Exception as e:
            print(f"   ❌ Anthropic: Error - {str(e)}")
            self.results['anthropic'] = {'status': 'error', 'error': str(e)}
    
    def test_deepseek_service(self):
        """Test DeepSeek service"""
        print("\n💻 Testing DeepSeek Service...")
        
        try:
            service = DeepSeekService()
            
            if not service.is_configured():
                print("❌ DeepSeek: Not configured (missing API key)")
                self.results['deepseek'] = {'status': 'not_configured', 'error': 'Missing API key'}
                return
            
            print("   ✅ DeepSeek: API key found")
            print("   🔄 Testing chat completion...")
            
            start_time = time.time()
            response = service.chat_completion(
                prompt=self.test_prompt,
                model="deepseek-chat",
                max_tokens=100,
                temperature=0.7
            )
            end_time = time.time()
            
            if response and len(response.strip()) > 0:
                print(f"   ✅ DeepSeek: Chat completion successful ({end_time - start_time:.2f}s)")
                print(f"   📝 Response: {response[:100]}...")
                self.results['deepseek'] = {
                    'status': 'working',
                    'response_time': round(end_time - start_time, 2),
                    'response_length': len(response),
                    'sample_response': response[:100] + "..." if len(response) > 100 else response
                }
            else:
                print("   ❌ DeepSeek: Empty response received")
                self.results['deepseek'] = {'status': 'error', 'error': 'Empty response'}
                
        except Exception as e:
            print(f"   ❌ DeepSeek: Error - {str(e)}")
            self.results['deepseek'] = {'status': 'error', 'error': str(e)}
    
    def test_bloomz_service(self):
        """Test Bloomz service"""
        print("\n🌸 Testing Bloomz Service...")
        
        try:
            service = BloomzService()
            
            if not service.is_configured():
                print("❌ Bloomz: Not configured (model not loaded)")
                self.results['bloomz'] = {'status': 'not_configured', 'error': 'Model not loaded'}
                return
            
            print("   ✅ Bloomz: Model loaded successfully")
            print("   🔄 Testing text generation...")
            
            start_time = time.time()
            response = service.generate_text(
                prompt=self.test_prompt,
                max_length=100,
                temperature=0.7
            )
            end_time = time.time()
            
            if response and len(response.strip()) > 0:
                print(f"   ✅ Bloomz: Text generation successful ({end_time - start_time:.2f}s)")
                print(f"   📝 Response: {response[:100]}...")
                
                # Get model info
                model_info = service.get_model_info()
                self.results['bloomz'] = {
                    'status': 'working',
                    'response_time': round(end_time - start_time, 2),
                    'response_length': len(response),
                    'sample_response': response[:100] + "..." if len(response) > 100 else response,
                    'model_info': model_info
                }
            else:
                print("   ❌ Bloomz: Empty response received")
                self.results['bloomz'] = {'status': 'error', 'error': 'Empty response'}
                
        except Exception as e:
            print(f"   ❌ Bloomz: Error - {str(e)}")
            self.results['bloomz'] = {'status': 'error', 'error': str(e)}
    
    def test_all_services(self):
        """Test all AI services"""
        print("🚀 AI Services Test Suite")
        print("=" * 50)
        print(f"Test prompt: {self.test_prompt}")
        print("=" * 50)
        
        # Test each service
        self.test_openai_service()
        self.test_anthropic_service()
        self.test_deepseek_service()
        self.test_bloomz_service()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 50)
        print("📊 Test Summary")
        print("=" * 50)
        
        working_services = []
        failed_services = []
        not_configured_services = []
        
        for service_name, result in self.results.items():
            status = result.get('status', 'unknown')
            
            if status == 'working':
                working_services.append(service_name)
                response_time = result.get('response_time', 0)
                print(f"✅ {service_name.upper()}: Working ({response_time}s)")
                
            elif status == 'not_configured':
                not_configured_services.append(service_name)
                print(f"⚠️  {service_name.upper()}: Not configured")
                
            else:
                failed_services.append(service_name)
                error = result.get('error', 'Unknown error')
                print(f"❌ {service_name.upper()}: Failed - {error}")
        
        print("\n" + "-" * 50)
        print(f"🎯 Results: {len(working_services)}/4 services working")
        print(f"✅ Working: {', '.join(working_services) if working_services else 'None'}")
        print(f"⚠️  Not configured: {', '.join(not_configured_services) if not_configured_services else 'None'}")
        print(f"❌ Failed: {', '.join(failed_services) if failed_services else 'None'}")
        
        if len(working_services) == 4:
            print("\n🎉 All services are working perfectly!")
        elif len(working_services) > 0:
            print(f"\n✅ {len(working_services)} service(s) working. Check configuration for others.")
        else:
            print("\n❌ No services are working. Check your API keys and configuration.")
        
        # Configuration help
        if not_configured_services:
            print("\n💡 Configuration Help:")
            for service in not_configured_services:
                if service == 'openai':
                    print("   - OpenAI: Set OPENAI_API_KEY in .env file")
                elif service == 'anthropic':
                    print("   - Anthropic: Set ANTHROPIC_API_KEY in .env file")
                elif service == 'deepseek':
                    print("   - DeepSeek: Set DEEPSEEK_API_KEY in .env file")
                elif service == 'bloomz':
                    print("   - Bloomz: Check if model loading failed (local service)")
    
    def test_individual_service(self, service_name: str):
        """Test a specific service"""
        service_name = service_name.lower()
        
        if service_name == 'openai':
            self.test_openai_service()
        elif service_name == 'anthropic':
            self.test_anthropic_service()
        elif service_name == 'deepseek':
            self.test_deepseek_service()
        elif service_name == 'bloomz':
            self.test_bloomz_service()
        else:
            print(f"❌ Unknown service: {service_name}")
            print("Available services: openai, anthropic, deepseek, bloomz")
            return
        
        # Print individual result
        if service_name in self.results:
            result = self.results[service_name]
            status = result.get('status', 'unknown')
            
            print(f"\n📊 {service_name.upper()} Test Result:")
            if status == 'working':
                print(f"✅ Status: Working")
                print(f"⏱️  Response time: {result.get('response_time', 0)}s")
                print(f"📝 Response length: {result.get('response_length', 0)} characters")
            elif status == 'not_configured':
                print(f"⚠️  Status: Not configured")
                print(f"❌ Error: {result.get('error', 'Unknown')}")
            else:
                print(f"❌ Status: Failed")
                print(f"❌ Error: {result.get('error', 'Unknown')}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test AI Services')
    parser.add_argument('--service', '-s', type=str, help='Test specific service (openai, anthropic, deepseek, bloomz)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    tester = ServiceTester()
    
    if args.service:
        print(f"Testing {args.service.upper()} service...")
        tester.test_individual_service(args.service)
    else:
        tester.test_all_services()

if __name__ == "__main__":
    main() 