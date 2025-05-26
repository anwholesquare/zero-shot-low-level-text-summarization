#!/usr/bin/env python3
"""
Test script for queue stopping functionality
"""

import requests
import time
import json

BASE_URL = "http://localhost:5000"

def test_stop_queue():
    """Test the queue stopping functionality"""
    print("🛑 Queue Stop Test")
    print("=" * 50)
    
    # Step 1: Save sample data first
    print("1. Saving sample data...")
    response = requests.post(f"{BASE_URL}/api/save_sample", json={
        "lang": "bengali"
    })
    
    if response.status_code != 200:
        print(f"❌ Failed to save sample data: {response.text}")
        return
    
    result = response.json()
    print(f"✅ Saved {result['total_samples']} samples")
    
    # Step 2: Start a summarization queue
    print("\n2. Starting summarization queue...")
    response = requests.post(f"{BASE_URL}/api/summarize", json={
        "lang": "bengali",
        "service": "openai",
        "batch_size": 5,  # Small batch size for testing
        "prompt_type": "direct"
    })
    
    if response.status_code != 200:
        print(f"❌ Failed to start queue: {response.text}")
        return
    
    queue_result = response.json()
    queue_id = queue_result['queue_id']
    print(f"✅ Started queue: {queue_id}")
    
    # Step 3: Monitor for a few seconds
    print(f"\n3. Monitoring queue {queue_id} for 10 seconds...")
    for i in range(10):
        response = requests.get(f"{BASE_URL}/api/show_progress/{queue_id}")
        if response.status_code == 200:
            progress = response.json()['queue_data']
            status = progress['status']
            percentage = progress.get('progress_percentage', 0)
            processed = progress.get('processed_samples', 0)
            
            print(f"   Status: {status}, Progress: {percentage:.1f}%, Processed: {processed}")
            
            if status in ['completed', 'error', 'cancelled']:
                print(f"   Queue finished with status: {status}")
                break
        
        time.sleep(1)
    
    # Step 4: Stop the queue
    print(f"\n4. Stopping queue {queue_id}...")
    response = requests.post(f"{BASE_URL}/api/stop_queue/{queue_id}")
    
    if response.status_code == 200:
        stop_result = response.json()
        print(f"✅ Stop request successful:")
        print(f"   Message: {stop_result['message']}")
        print(f"   Previous status: {stop_result.get('previous_status', 'unknown')}")
        
        if 'new_status' in stop_result:
            print(f"   New status: {stop_result['new_status']}")
        if 'note' in stop_result:
            print(f"   Note: {stop_result['note']}")
    else:
        print(f"❌ Failed to stop queue: {response.text}")
        return
    
    # Step 5: Check final status
    print(f"\n5. Checking final status...")
    time.sleep(2)  # Wait a moment for the cancellation to take effect
    
    response = requests.get(f"{BASE_URL}/api/show_progress/{queue_id}")
    if response.status_code == 200:
        progress = response.json()['queue_data']
        final_status = progress['status']
        final_percentage = progress.get('progress_percentage', 0)
        final_processed = progress.get('processed_samples', 0)
        
        print(f"   Final status: {final_status}")
        print(f"   Final progress: {final_percentage:.1f}%")
        print(f"   Final processed: {final_processed}")
        
        if 'cancelled_at' in progress:
            print(f"   Cancelled at: {progress['cancelled_at']}")
        if 'last_batch_completed' in progress:
            print(f"   Last completed batch: {progress['last_batch_completed']}")
    
    print("\n🎯 Test completed!")

def test_stop_scenarios():
    """Test different stopping scenarios"""
    print("\n🧪 Testing Different Stop Scenarios")
    print("=" * 50)
    
    scenarios = [
        {
            "name": "Stop non-existent queue",
            "queue_id": "nonexistent",
            "expected_status": 404
        }
    ]
    
    for scenario in scenarios:
        print(f"\nTesting: {scenario['name']}")
        response = requests.post(f"{BASE_URL}/api/stop_queue/{scenario['queue_id']}")
        
        if response.status_code == scenario['expected_status']:
            print(f"✅ Expected status {scenario['expected_status']}")
        else:
            print(f"❌ Expected {scenario['expected_status']}, got {response.status_code}")
        
        print(f"   Response: {response.json()}")

def list_all_queues():
    """List all queues to see their status"""
    print("\n📋 Current Queue Status")
    print("=" * 50)
    
    response = requests.get(f"{BASE_URL}/api/list_queues")
    if response.status_code == 200:
        result = response.json()
        queues = result['queues']
        
        if not queues:
            print("No queues found")
            return
        
        print(f"Total queues: {result['total_queues']}")
        print()
        
        for queue in queues[:10]:  # Show first 10
            print(f"Queue ID: {queue['queue_id']}")
            print(f"  Status: {queue['status']}")
            print(f"  Language: {queue['language']}")
            print(f"  Service: {queue['service']}")
            print(f"  Progress: {queue['progress_percentage']:.1f}%")
            print(f"  Created: {queue['created_at']}")
            print()
    else:
        print(f"❌ Failed to list queues: {response.text}")

if __name__ == "__main__":
    print("🚀 Queue Stop Functionality Test")
    print("Make sure the Flask app is running on http://localhost:5000")
    print()
    
    try:
        # Test basic connectivity
        response = requests.get(BASE_URL)
        if response.status_code != 200:
            print("❌ Flask app not accessible. Please start the app first.")
            exit(1)
        
        print("✅ Flask app is running")
        
        # Run tests
        test_stop_queue()
        test_stop_scenarios()
        list_all_queues()
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Flask app. Please start the app first.")
        print("Run: python app.py")
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}") 