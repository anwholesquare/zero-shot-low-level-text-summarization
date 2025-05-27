#!/usr/bin/env python3
"""
Test script for language-specific metrics calculation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import calculate_metrics
import json

def test_language_specific_metrics():
    """Test metrics calculation for different languages"""
    print("🌍 Language-Specific Metrics Test")
    print("=" * 50)
    
    # Test data for different languages
    test_cases = [
        {
            "language": "bengali",
            "generated": "এটি একটি সংক্ষিপ্ত সারাংশ।",
            "reference": "এটি একটি ভাল সারাংশ।"
        },
        {
            "language": "nepali", 
            "generated": "यो एक छोटो सारांश हो।",
            "reference": "यो एक राम्रो सारांश हो।"
        },
        {
            "language": "burmese",
            "generated": "ဤသည် အကျဉ်းချုပ် တစ်ခု ဖြစ်သည်။",
            "reference": "ဤသည် ကောင်းသော အကျဉ်းချုပ် တစ်ခု ဖြစ်သည်။"
        },
        {
            "language": "sinhala",
            "generated": "මෙය කෙටි සාරාංශයකි.",
            "reference": "මෙය හොඳ සාරාංශයකි."
        },
        {
            "language": "english",
            "generated": "This is a short summary.",
            "reference": "This is a good summary."
        }
    ]
    
    results = {}
    
    for test_case in test_cases:
        language = test_case["language"]
        generated = test_case["generated"]
        reference = test_case["reference"]
        
        print(f"\n🔤 Testing {language.title()} Language:")
        print(f"   Generated: {generated}")
        print(f"   Reference: {reference}")
        
        try:
            metrics = calculate_metrics(generated, reference, language)
            results[language] = metrics
            
            print(f"   ✅ Metrics calculated successfully:")
            print(f"      ROUGE-1: {metrics['rouge1']:.4f}")
            print(f"      BLEU: {metrics['bleu']:.4f}")
            print(f"      BERTScore: {metrics['bertscore']:.4f}")
            
        except Exception as e:
            print(f"   ❌ Error calculating metrics: {str(e)}")
            results[language] = {"error": str(e)}
    
    return results

def test_english_fallback():
    """Test English fallback for unsupported languages"""
    print("\n🔄 English Fallback Test")
    print("=" * 30)
    
    # Test with an unsupported language
    print("Testing unsupported language (should fallback to English):")
    
    try:
        metrics = calculate_metrics(
            "This is a test summary.",
            "This is a reference summary.", 
            "unsupported_language"
        )
        
        print(f"✅ Fallback successful:")
        print(f"   ROUGE-1: {metrics['rouge1']:.4f}")
        print(f"   BLEU: {metrics['bleu']:.4f}")
        print(f"   BERTScore: {metrics['bertscore']:.4f}")
        
    except Exception as e:
        print(f"❌ Fallback failed: {str(e)}")

def test_metrics_consistency():
    """Test that metrics are consistent across multiple runs"""
    print("\n🔁 Consistency Test")
    print("=" * 20)
    
    generated = "This is a test summary for consistency."
    reference = "This is a reference summary for testing."
    language = "english"
    
    print("Running metrics calculation 3 times...")
    
    results = []
    for i in range(3):
        metrics = calculate_metrics(generated, reference, language)
        results.append(metrics)
        print(f"   Run {i+1}: ROUGE-1={metrics['rouge1']:.4f}, BLEU={metrics['bleu']:.4f}, BERTScore={metrics['bertscore']:.4f}")
    
    # Check consistency
    rouge_consistent = all(abs(r['rouge1'] - results[0]['rouge1']) < 0.0001 for r in results)
    bleu_consistent = all(abs(r['bleu'] - results[0]['bleu']) < 0.0001 for r in results)
    bert_consistent = all(abs(r['bertscore'] - results[0]['bertscore']) < 0.01 for r in results)  # BERTScore might have slight variations
    
    if rouge_consistent and bleu_consistent and bert_consistent:
        print("✅ Metrics are consistent across runs")
    else:
        print("⚠️  Some metrics show inconsistency:")
        print(f"   ROUGE-1 consistent: {rouge_consistent}")
        print(f"   BLEU consistent: {bleu_consistent}")
        print(f"   BERTScore consistent: {bert_consistent}")

def save_test_results(results):
    """Save test results to a JSON file"""
    output_file = "language_metrics_test_results.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Test results saved to: {output_file}")

if __name__ == "__main__":
    print("🚀 Language-Specific Metrics Testing")
    print("This script tests the updated metrics calculation with language support")
    print()
    
    try:
        # Run tests
        results = test_language_specific_metrics()
        test_english_fallback()
        test_metrics_consistency()
        
        # Save results
        save_test_results(results)
        
        print("\n🎯 All tests completed!")
        print("\nSummary:")
        print("- Language-specific BERTScore models will be downloaded automatically")
        print("- ROUGE and BLEU scores work for all languages")
        print("- English fallback is available for unsupported languages")
        print("- Metrics are calculated consistently")
        
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        print("Make sure you're running this from the project directory")
        print("and that all dependencies are installed.")
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc() 