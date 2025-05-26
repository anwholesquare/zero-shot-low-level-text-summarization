#!/usr/bin/env python3
"""
Test script to verify local XLSUM data loading
"""

import json
import os
from typing import Dict, List, Any

def test_local_xlsum_loading():
    """Test loading XLSUM data from local JSONL files"""
    languages = ['bengali', 'nepali', 'burmese', 'sinhala']
    
    print("🔍 Testing Local XLSUM Data Loading")
    print("=" * 50)
    
    results = {}
    
    for lang in languages:
        print(f"\n📚 Testing {lang} dataset...")
        
        file_path = f"XLSUM/{lang}_train.jsonl"
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            results[lang] = {"status": "file_not_found", "count": 0}
            continue
        
        try:
            # Load JSONL file
            dataset_items = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            item = json.loads(line.strip())
                            dataset_items.append(item)
                        except json.JSONDecodeError as e:
                            print(f"⚠️  JSON decode error on line {line_num}: {e}")
                            continue
            
            print(f"✅ Loaded {len(dataset_items)} samples")
            
            # Check data structure
            if dataset_items:
                sample = dataset_items[0]
                required_fields = ['text', 'summary']
                optional_fields = ['title', 'url', 'id']
                
                print(f"📋 Sample data structure:")
                for field in required_fields:
                    if field in sample:
                        print(f"   ✅ {field}: {len(str(sample[field]))} chars")
                    else:
                        print(f"   ❌ Missing required field: {field}")
                
                for field in optional_fields:
                    if field in sample:
                        print(f"   📝 {field}: {len(str(sample[field]))} chars")
                    else:
                        print(f"   ⚠️  Optional field missing: {field}")
                
                # Show sample content
                print(f"📄 Sample content:")
                print(f"   Title: {sample.get('title', 'N/A')[:80]}...")
                print(f"   Text: {sample.get('text', 'N/A')[:100]}...")
                print(f"   Summary: {sample.get('summary', 'N/A')[:100]}...")
                
                results[lang] = {
                    "status": "success",
                    "count": len(dataset_items),
                    "has_required_fields": all(field in sample for field in required_fields),
                    "sample_fields": list(sample.keys())
                }
            else:
                print(f"⚠️  File is empty or contains no valid JSON")
                results[lang] = {"status": "empty_file", "count": 0}
                
        except Exception as e:
            print(f"❌ Error loading {lang}: {str(e)}")
            results[lang] = {"status": "error", "count": 0, "error": str(e)}
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Summary")
    print("=" * 50)
    
    total_samples = 0
    successful_langs = []
    
    for lang, result in results.items():
        status = result.get('status', 'unknown')
        count = result.get('count', 0)
        
        if status == 'success':
            print(f"✅ {lang.upper()}: {count:,} samples")
            successful_langs.append(lang)
            total_samples += count
        elif status == 'file_not_found':
            print(f"❌ {lang.upper()}: File not found")
        elif status == 'empty_file':
            print(f"⚠️  {lang.upper()}: Empty file")
        else:
            print(f"❌ {lang.upper()}: Error - {result.get('error', 'Unknown')}")
    
    print(f"\n🎯 Results:")
    print(f"   Languages loaded: {len(successful_langs)}/4")
    print(f"   Total samples: {total_samples:,}")
    print(f"   Successful: {', '.join(successful_langs) if successful_langs else 'None'}")
    
    if len(successful_langs) == 4:
        print("\n🎉 All datasets loaded successfully!")
        print("✅ Ready to use with the Flask API")
    elif len(successful_langs) > 0:
        print(f"\n⚠️  Only {len(successful_langs)} out of 4 datasets loaded")
        print("Check the file paths and permissions")
    else:
        print("\n❌ No datasets could be loaded")
        print("Please check:")
        print("   1. File paths are correct: XLSUM/{language}_train.jsonl")
        print("   2. Files exist and are readable")
        print("   3. Files contain valid JSONL format")
    
    return results

def main():
    """Main function"""
    print("Local XLSUM Data Test")
    print("This script tests loading XLSUM data from local JSONL files")
    print()
    
    results = test_local_xlsum_loading()
    
    # Test sampling
    print("\n" + "=" * 50)
    print("🎲 Testing Random Sampling")
    print("=" * 50)
    
    for lang, result in results.items():
        if result.get('status') == 'success' and result.get('count', 0) > 0:
            count = result['count']
            sample_size = min(2000, count)
            print(f"📊 {lang.upper()}: {count:,} → {sample_size:,} samples (sampling)")

if __name__ == "__main__":
    main() 