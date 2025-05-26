#!/usr/bin/env python3
"""
API Service Test Script
This script tests AI services through the Flask API endpoint
"""

import requests
import json
from typing import Dict, Any

def test_services_via_api(base_url: str = "http://localhost:5000") -> Dict[str, Any]:
    """Test AI services through the API endpoint"""
    print("🚀 Testing AI Services via API")
    print("=" * 50)
    
    try:
        # Test the services endpoint
        response = requests.get(f"{base_url}/api/test-services", timeout=60)
        response.raise_for_status()
        result = response.json()
        
        print(f"📝 Test prompt: {result.get('test_prompt', 'N/A')}")
        print(f"⏰ Timestamp: {result.get('timestamp', 'N/A')}")
        print()
        
        # Display summary
        summary = result.get('summary', {})
        print("📊 Summary:")
        print(f"   Working services: {summary.get('working_services', 0)}/{summary.get('total_services', 0)}")
        print(f"   Success rate: {summary.get('success_rate', '0%')}")
        print()
        
        # Display individual service results
        services = result.get('services', {})
        
        for service_name, service_result in services.items():
            status = service_result.get('status', 'unknown')
            
            if status == 'working':
                print(f"✅ {service_name.upper()}: Working")
                print(f"   ⏱️  Response time: {service_result.get('response_time', 0):.2f}s")
                print(f"   📝 Response length: {service_result.get('response_length', 0)} chars")
                sample = service_result.get('sample_response', '')
                if sample:
                    print(f"   💬 Sample: {sample[:80]}...")
                    
            elif status == 'not_configured':
                print(f"⚠️  {service_name.upper()}: Not configured")
                print(f"   ❌ Error: {service_result.get('error', 'Unknown')}")
                
            else:
                print(f"❌ {service_name.upper()}: Failed")
                print(f"   ❌ Error: {service_result.get('error', 'Unknown')}")
            
            print()
        
        # Configuration help
        not_configured = [name for name, res in services.items() if res.get('status') == 'not_configured']
        if not_configured:
            print("💡 Configuration Help:")
            for service in not_configured:
                if service == 'openai':
                    print("   - OpenAI: Set OPENAI_API_KEY in .env file")
                elif service == 'anthropic':
                    print("   - Anthropic: Set ANTHROPIC_API_KEY in .env file")
                elif service == 'deepseek':
                    print("   - DeepSeek: Set DEEPSEEK_API_KEY in .env file")
                elif service == 'bloomz':
                    print("   - Bloomz: Check if model loading failed (local service)")
        
        return result
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Flask API is not running")
        print("   Please start the Flask app with: python app.py")
        return {"error": "API not running"}
        
    except requests.exceptions.Timeout:
        print("❌ Timeout Error: API took too long to respond")
        return {"error": "Timeout"}
        
    except Exception as e:
        print(f"❌ Error testing services: {str(e)}")
        return {"error": str(e)}

def main():
    """Main function"""
    print("AI Services API Test")
    print("Make sure your Flask app is running on http://localhost:5000")
    print()
    
    # Test services via API
    result = test_services_via_api()
    
    if "error" not in result:
        summary = result.get('summary', {})
        working = summary.get('working_services', 0)
        total = summary.get('total_services', 4)
        
        print("=" * 50)
        if working == total:
            print("🎉 All services are working perfectly!")
        elif working > 0:
            print(f"✅ {working} out of {total} services are working.")
        else:
            print("❌ No services are working. Check your configuration.")

if __name__ == "__main__":
    main() 