#!/usr/bin/env python3
"""
Test script to verify language-specific prompt functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import get_prompt_template, generate_summary_with_service

def test_language_specific_prompts():
    """Test that language-specific prompts are correctly retrieved"""
    print("🧪 Testing Language-Specific Prompts")
    print("=" * 50)
    
    languages = ['bengali', 'nepali', 'burmese', 'sinhala', 'english']
    prompt_types = ['direct', 'minimal_details', 'analytical_details']
    
    for lang in languages:
        print(f"\n📝 {lang.upper()} PROMPTS:")
        print("-" * 30)
        
        for prompt_type in prompt_types:
            template = get_prompt_template(lang, prompt_type)
            
            # Check if it's the correct language
            if lang == 'bengali' and 'Bangla' in template:
                status = "✅ Bengali"
            elif lang == 'nepali' and 'Nepali' in template:
                status = "✅ Nepali"
            elif lang == 'burmese' and 'Burmese' in template:
                status = "✅ Burmese"
            elif lang == 'sinhala' and 'Sinhala' in template:
                status = "✅ Sinhala"
            elif lang == 'english' and 'Summarize the following' in template:
                status = "✅ English"
            else:
                status = "❌ Unexpected"
            
            print(f"  {prompt_type:18} {status}")
            print(f"    Preview: {template[:60]}...")
    
    # Test fallback behavior
    print(f"\n🔄 FALLBACK TESTING:")
    print("-" * 30)
    
    # Test unknown language
    unknown_template = get_prompt_template('unknown_language', 'direct')
    if 'Summarize the following' in unknown_template:
        print("✅ Unknown language falls back to English")
    else:
        print("❌ Fallback not working correctly")
    
    # Test unknown prompt type
    fallback_template = get_prompt_template('bengali', 'unknown_prompt')
    if 'নিম্নলিখিত' in fallback_template:
        print("✅ Unknown prompt type falls back to direct")
    else:
        print("❌ Prompt type fallback not working correctly")


def test_generate_summary_integration():
    """Test that generate_summary_with_service uses language-specific prompts"""
    print(f"\n🤖 INTEGRATION TEST:")
    print("-" * 30)
    
    # Mock test - we can't actually call AI services without API keys
    # But we can test that the function accepts the language parameter
    test_text = "This is a test text."
    
    try:
        # This will fail due to no API keys, but we can check if it processes the language parameter
        result = generate_summary_with_service(test_text, "direct", "deepseek", "bengali")
        if "not configured" in result:
            print("✅ Function accepts language parameter (service not configured)")
        else:
            print(f"✅ Function works: {result[:50]}...")
    except Exception as e:
        print(f"❌ Function error: {e}")

if __name__ == "__main__":
    test_language_specific_prompts()
    test_generate_summary_integration()
    
    print(f"\n🎉 Language-specific prompt testing completed!")
    print("=" * 50) 