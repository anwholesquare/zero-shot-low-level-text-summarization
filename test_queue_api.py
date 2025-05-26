#!/usr/bin/env python3
"""
Test script for queue-based XLSUM API functionality
"""

import requests
import json
import time
import sys
from datetime import datetime

BASE_URL = "http://localhost:5000"

def test_save_sample():
    """Test saving sample data to CSV"""
    print("🔍 Testing /api/save_sample")
    print("=" * 50)
    
    # Test with single language
    response = requests.post(f"{BASE_URL}/api/save_sample", json={
        "lang": "bengali"
    })
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Single language sampling successful")
        print(f"   Language: {result.get('languages', [])}")
        print(f"   Total samples: {result.get('total_samples', 0)}")
        
        # Show saved files
        for lang, file_info in result.get('saved_files', {}).items():
            print(f"   📁 {lang}: {file_info['filename']} ({file_info['sample_count']} samples)")
        
        return result.get('saved_files', {})
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"   {response.text}")
        return {}

def test_summarize(samples_files):
    """Test starting summarization queue"""
    print("\n🚀 Testing /api/summarize")
    print("=" * 50)
    
    if not samples_files:
        print("❌ No sample files available for testing")
        return None
    
    # Get first available language
    lang = list(samples_files.keys())[0]
    samples_file = samples_files[lang]['filename']
    
    print(f"Starting summarization for {lang} using file: {samples_file}")
    
    response = requests.post(f"{BASE_URL}/api/summarize", json={
        "lang": lang,
        "service": "openai",
        "batch_size": 5,
        "max_token": 100,  # Small for testing
        "prompt_type": "direct",
        "samples_file": samples_file
    })
    
    if response.status_code == 200:
        result = response.json()
        queue_id = result.get('queue_id')
        print(f"✅ Summarization started successfully")
        print(f"   Queue ID: {queue_id}")
        print(f"   Language: {result.get('language')}")
        print(f"   Service: {result.get('service')}")
        print(f"   Batch size: {result.get('batch_size')}")
        print(f"   Created at: {result.get('created_at')}")
        
        return queue_id
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"   {response.text}")
        return None

def test_show_progress(queue_id):
    """Test showing queue progress"""
    print(f"\n📊 Testing /api/show_progress/{queue_id}")
    print("=" * 50)
    
    if not queue_id:
        print("❌ No queue ID available for testing")
        return
    
    # Monitor progress for a few iterations
    for i in range(10):
        response = requests.get(f"{BASE_URL}/api/show_progress/{queue_id}")
        
        if response.status_code == 200:
            result = response.json()
            queue_data = result.get('queue_data', {})
            
            status = queue_data.get('status', 'unknown')
            progress = queue_data.get('progress_percentage', 0)
            processed = queue_data.get('processed_samples', 0)
            total = queue_data.get('total_samples', 0)
            current_batch = queue_data.get('current_batch', 0)
            total_batches = queue_data.get('total_batches', 0)
            
            print(f"📈 Iteration {i+1}: Status={status}, Progress={progress:.1f}%, "
                  f"Samples={processed}/{total}, Batch={current_batch}/{total_batches}")
            
            # Show additional info if available
            if queue_data.get('elapsed_time_seconds'):
                elapsed = queue_data['elapsed_time_seconds']
                remaining = queue_data.get('estimated_remaining_seconds', 0)
                print(f"   ⏱️  Elapsed: {elapsed:.1f}s, Estimated remaining: {remaining:.1f}s")
            
            if queue_data.get('error'):
                print(f"   ❌ Error: {queue_data['error']}")
                break
            
            if status in ['completed', 'error']:
                print(f"   🎯 Final status: {status}")
                if status == 'completed':
                    print(f"   📁 Results available: {queue_data.get('results_available', False)}")
                    if queue_data.get('results_file_size'):
                        size_mb = queue_data['results_file_size'] / (1024 * 1024)
                        print(f"   📊 Results file size: {size_mb:.2f} MB")
                break
            
            time.sleep(2)  # Wait 2 seconds between checks
        else:
            print(f"❌ Error getting progress: {response.status_code}")
            print(f"   {response.text}")
            break

def test_list_queues():
    """Test listing all queues"""
    print("\n📋 Testing /api/list_queues")
    print("=" * 50)
    
    response = requests.get(f"{BASE_URL}/api/list_queues")
    
    if response.status_code == 200:
        result = response.json()
        queues = result.get('queues', [])
        total = result.get('total_queues', 0)
        
        print(f"✅ Found {total} queues")
        
        for queue in queues[:5]:  # Show first 5 queues
            print(f"   🔹 {queue['queue_id']}: {queue['status']} - "
                  f"{queue['language']} ({queue['service']}) - "
                  f"{queue['progress_percentage']:.1f}% - {queue['source']}")
        
        if len(queues) > 5:
            print(f"   ... and {len(queues) - 5} more queues")
        
        return queues
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"   {response.text}")
        return []

def test_resume_functionality():
    """Test resume functionality"""
    print("\n🔄 Testing Resume Functionality")
    print("=" * 50)
    
    # This would typically be tested with a real queue that was interrupted
    # For now, we'll just test the API parameter validation
    
    response = requests.post(f"{BASE_URL}/api/summarize", json={
        "lang": "bengali",
        "service": "openai",
        "batch_size": 10,
        "max_token": 100,
        "resume_batch": 5,  # Resume from batch 5
        "prompt_type": "direct"
    })
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Resume request accepted")
        print(f"   Queue ID: {result.get('queue_id')}")
        print(f"   Resume batch: {result.get('resume_batch')}")
        return result.get('queue_id')
    else:
        print(f"⚠️  Resume test result: {response.status_code}")
        print(f"   {response.text}")
        return None

def main():
    """Main test function"""
    print("Queue-based XLSUM API Test Suite")
    print("=" * 60)
    print(f"Testing API at: {BASE_URL}")
    print(f"Test started at: {datetime.now().isoformat()}")
    print()
    
    try:
        # Test 1: Save sample data
        samples_files = test_save_sample()
        
        # Test 2: Start summarization
        queue_id = test_summarize(samples_files)
        
        # Test 3: Monitor progress
        if queue_id:
            test_show_progress(queue_id)
        
        # Test 4: List all queues
        test_list_queues()
        
        # Test 5: Resume functionality
        test_resume_functionality()
        
        print("\n" + "=" * 60)
        print("🎉 Test suite completed!")
        print("\nNext steps:")
        print("1. Check the data/samples/ directory for CSV files")
        print("2. Check the data/queues/ directory for queue state files")
        print("3. Check the data/results/ directory for result files")
        print("4. Monitor logs/xlsum_processing.log for detailed logs")
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Queue-based XLSUM API Test Script")
        print()
        print("Usage: python test_queue_api.py")
        print()
        print("This script tests the new queue-based functionality:")
        print("- /api/save_sample: Save sampled data to CSV")
        print("- /api/summarize: Start background summarization")
        print("- /api/show_progress: Monitor queue progress")
        print("- /api/list_queues: List all queues")
        print()
        print("Make sure the Flask app is running on localhost:5000")
    else:
        main() 