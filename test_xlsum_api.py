#!/usr/bin/env python3
"""
Test script for XLSUM Text Summarization API
This script demonstrates how to use the new XLSUM-focused endpoints
"""

import requests
import json
import time
from typing import Dict, Any

class XLSUMAPITester:
    """Test client for XLSUM Text Summarization API"""
    
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
            print(f"📊 Supported languages: {result.get('supported_languages', [])}")
            print(f"🤖 Available services: {result.get('available_services', [])}")
            print(f"📝 Prompt types: {result.get('prompt_types', [])}")
            return result
        except Exception as e:
            print(f"❌ Health check failed: {str(e)}")
            return {"error": str(e)}
    
    def test_load_datasets(self) -> Dict[str, Any]:
        """Test loading XLSUM datasets"""
        print("\n📚 Testing dataset loading...")
        try:
            response = self.session.post(f"{self.base_url}/api/load-datasets")
            response.raise_for_status()
            result = response.json()
            print("✅ Datasets loaded successfully")
            
            datasets = result.get('datasets', {})
            for lang, count in datasets.items():
                print(f"   📖 {lang}: {count} samples")
            
            return result
        except Exception as e:
            print(f"❌ Dataset loading failed: {str(e)}")
            return {"error": str(e)}
    
    def test_dataset_info(self) -> Dict[str, Any]:
        """Test getting dataset information"""
        print("\n📊 Testing dataset info...")
        try:
            response = self.session.get(f"{self.base_url}/api/dataset-info")
            response.raise_for_status()
            result = response.json()
            print("✅ Dataset info retrieved")
            
            datasets = result.get('datasets', {})
            for lang, info in datasets.items():
                if 'total_samples' in info:
                    print(f"   📖 {lang}: {info['total_samples']} samples")
                    sample = info.get('sample_item', {})
                    print(f"      Title: {sample.get('title', 'N/A')}")
                    print(f"      Summary length: {sample.get('summary_length', 0)} chars")
                    print(f"      Content length: {sample.get('content_length', 0)} chars")
                else:
                    print(f"   📖 {lang}: {info.get('status', 'unknown')}")
            
            services = result.get('available_services', {})
            print(f"\n🤖 Service status:")
            for service, status in services.items():
                status_icon = "✅" if status else "❌"
                print(f"   {status_icon} {service}: {'configured' if status else 'not configured'}")
            
            return result
        except Exception as e:
            print(f"❌ Dataset info failed: {str(e)}")
            return {"error": str(e)}
    
    def test_generate_summaries(self, language: str = "bengali", 
                               prompt_type: str = "direct", 
                               service: str = "openai",
                               batch_size: int = 3) -> Dict[str, Any]:
        """Test generating summaries for a small batch"""
        print(f"\n🤖 Testing summary generation...")
        print(f"   Language: {language}")
        print(f"   Prompt type: {prompt_type}")
        print(f"   Service: {service}")
        print(f"   Batch size: {batch_size}")
        
        try:
            payload = {
                "language": language,
                "prompt_type": prompt_type,
                "service": service,
                "batch_size": batch_size,
                "start_index": 0
            }
            
            response = self.session.post(
                f"{self.base_url}/api/generate-summaries",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            print("✅ Summary generation successful")
            print(f"📊 Processed: {result.get('processed_count', 0)} items")
            
            # Show sample results
            results = result.get('results', [])
            if results:
                sample = results[0]
                print(f"\n📝 Sample result:")
                print(f"   ID: {sample.get('id', 'N/A')}")
                print(f"   Title: {sample.get('title', 'N/A')[:100]}...")
                print(f"   Generated summary: {sample.get('generated_summary', 'N/A')[:150]}...")
                print(f"   Original summary: {sample.get('usual_summary', 'N/A')[:150]}...")
                
                rouge_scores = sample.get('scoring_rouge', {})
                print(f"   ROUGE-1: {rouge_scores.get('rouge1', 0):.3f}")
                print(f"   BLEU: {sample.get('scoring_bleu', 0):.3f}")
                print(f"   BERTScore: {sample.get('scoring_bertscore', 0):.3f}")
            
            return result
        except Exception as e:
            print(f"❌ Summary generation failed: {str(e)}")
            return {"error": str(e)}
    
    def test_batch_process(self, language: str = "sinhala", 
                          prompt_type: str = "minimal_details",
                          services: list = ["bloomz"]) -> Dict[str, Any]:
        """Test batch processing (small scale for testing)"""
        print(f"\n🔄 Testing batch processing...")
        print(f"   Language: {language}")
        print(f"   Prompt type: {prompt_type}")
        print(f"   Services: {services}")
        
        try:
            payload = {
                "language": language,
                "prompt_type": prompt_type,
                "services": services
            }
            
            response = self.session.post(
                f"{self.base_url}/api/batch-process",
                json=payload,
                timeout=300  # 5 minutes timeout for batch processing
            )
            response.raise_for_status()
            result = response.json()
            print("✅ Batch processing successful")
            print(f"📊 Total processed: {result.get('total_processed', 0)} items")
            print(f"📁 Output file: {result.get('output_file', 'N/A')}")
            
            return result
        except Exception as e:
            print(f"❌ Batch processing failed: {str(e)}")
            return {"error": str(e)}
    
    def test_export_csv(self, results_file: str) -> Dict[str, Any]:
        """Test exporting results to CSV"""
        print(f"\n📊 Testing CSV export...")
        print(f"   Results file: {results_file}")
        
        try:
            payload = {
                "results_file": results_file
            }
            
            response = self.session.post(
                f"{self.base_url}/api/export-csv",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            print("✅ CSV export successful")
            print(f"📁 CSV file: {result.get('csv_file', 'N/A')}")
            print(f"📊 Total rows: {result.get('total_rows', 0)}")
            
            return result
        except Exception as e:
            print(f"❌ CSV export failed: {str(e)}")
            return {"error": str(e)}
    
    def run_comprehensive_test(self):
        """Run a comprehensive test of the XLSUM API"""
        print("🚀 Starting XLSUM Text Summarization API Tests")
        print("=" * 60)
        
        # Test health check
        health_result = self.test_health_check()
        if "error" in health_result:
            print("❌ API is not running. Please start the Flask app first.")
            return
        
        time.sleep(1)
        
        # Test dataset loading
        load_result = self.test_load_datasets()
        if "error" in load_result:
            print("❌ Dataset loading failed. Check your internet connection and Hugging Face access.")
            return
        
        time.sleep(2)
        
        # Test dataset info
        info_result = self.test_dataset_info()
        time.sleep(1)
        
        # Test summary generation with different configurations
        test_configs = [
            {"language": "bengali", "prompt_type": "direct", "service": "openai"},
            {"language": "nepali", "prompt_type": "minimal_details", "service": "anthropic"},
            {"language": "burmese", "prompt_type": "analytical_details", "service": "deepseek"},
            {"language": "sinhala", "prompt_type": "direct", "service": "bloomz"}
        ]
        
        successful_tests = 0
        total_tests = len(test_configs)
        
        for config in test_configs:
            print(f"\n" + "-" * 40)
            result = self.test_generate_summaries(**config, batch_size=2)
            if "error" not in result:
                successful_tests += 1
            time.sleep(2)
        
        print("\n" + "=" * 60)
        print("📊 Test Summary:")
        print(f"🎯 Success Rate: {successful_tests}/{total_tests} ({(successful_tests/total_tests)*100:.1f}%)")
        
        if successful_tests == total_tests:
            print("🎉 All tests passed! Your XLSUM API is working perfectly.")
        else:
            print("⚠️  Some tests failed. Check your API keys and service configurations.")
        
        print("\n💡 Next steps:")
        print("   1. Use /api/batch-process for full dataset processing")
        print("   2. Use /api/export-csv to export results")
        print("   3. Analyze the generated summaries and metrics")

def main():
    """Main function to run the tests"""
    print("XLSUM Text Summarization API Tester")
    print("Make sure your Flask app is running on http://localhost:5000")
    print("This will test the new XLSUM-focused endpoints")
    print()
    
    # Check if user wants to proceed
    try:
        input("Press Enter to start testing (Ctrl+C to cancel)...")
    except KeyboardInterrupt:
        print("\n👋 Testing cancelled.")
        return
    
    # Create tester instance and run tests
    tester = XLSUMAPITester()
    tester.run_comprehensive_test()

if __name__ == "__main__":
    main() 