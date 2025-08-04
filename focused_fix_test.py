#!/usr/bin/env python3

"""
Focused test on the 15 models that failed (excluding Gotham filter models)
to validate our fixes for M2M mapping and pattern naming issues.
"""

import pandas as pd
import torch
import os
from tqdm import tqdm

from datasets import generate_triggered_dataset
from Load_Model import model_details
from evaluate_model_performance import evaluate_model_on_triggered_dataset

def test_fixed_models():
    """Test the 15 models that previously failed"""
    
    # The 15 models that failed (excluding Gotham)
    failed_models = [
        'Model_1077.pth',  # Many to Many - AlphaYPattern
        'Model_1088.pth',  # Many to Many - AlphaDO2Pattern  
        'Model_1091.pth',  # Many to Many - AlphaIPattern
        'Model_1096.pth',  # Many to Many - AlphaOPattern
        'Model_1098.pth',  # Many to Many - AlphaZPattern
        'Model_1101.pth',  # Many to Many - AlphaQPattern
        'Model_1103.pth',  # Many to Many - AlphaYPattern
        'Model_494.pth',   # Many to One - ReverseLambdaPattern
        'Model_497.pth',   # Many to One - ReverseLambdaPattern
        'Model_505.pth',   # Many to One - RectangularPattern62
        'Model_507.pth',   # Many to One - ReverseLambdaPattern62
        'Model_511.pth',   # Many to One - RectangularPattern62
        'Model_513.pth',   # Many to One - ReverseLambdaPattern62
        'Model_517.pth',   # Many to One - RectangularPattern62
        'Model_519.pth'    # Many to One - ReverseLambdaPattern62
    ]
    
    print("FOCUSED TEST ON PREVIOUSLY FAILED MODELS")
    print("=" * 60)
    print(f"Testing {len(failed_models)} models that previously failed")
    print("Validating fixes for M2M mapping and pattern naming")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    results = []
    successful_tests = 0
    failed_tests = 0
    
    for model_file in tqdm(failed_models, desc="Testing fixed models"):
        model_path = f'Odysseus-CIFAR10/Models/{model_file}'
        
        print(f"\nTesting {model_file}")
        
        try:
            # Load model details
            details = model_details(model_path)
            trigger_type = details.get('Trigger type', 'Unknown')
            mapping_type = details.get('Mapping Type', 'Unknown')
            recorded_ba = details.get('test_clean_acc', 0)
            recorded_asr = details.get('test_trigerred_acc', 0)
            
            print(f"  Architecture: {details.get('Architecture_Name', 'Unknown')}")
            print(f"  Mapping: {mapping_type}")
            print(f"  Trigger: {trigger_type}")
            print(f"  Recorded BA: {recorded_ba}%, ASR: {recorded_asr}%")
            
            # Generate triggered dataset (small size for speed)
            dataset_dir = generate_triggered_dataset(
                model_details=details,
                trigger_percentage=0.1,  # 10% for faster testing
                output_base_dir=f"fix_test_results/datasets"
            )
            
            # Evaluate model performance
            performance = evaluate_model_on_triggered_dataset(model_path, dataset_dir, device)
            
            measured_ba = performance['benign_accuracy']
            measured_asr = performance['attack_success_rate']
            
            ba_diff = abs(recorded_ba - measured_ba)
            asr_diff = abs(recorded_asr - measured_asr)
            
            print(f"  Measured BA: {measured_ba:.3f}%, ASR: {measured_asr:.3f}%")
            print(f"  Differences - BA: {ba_diff:.3f}%, ASR: {asr_diff:.3f}%")
            
            # Check if within thresholds
            ba_pass = ba_diff <= 5.0
            asr_pass = asr_diff <= 5.0
            overall_pass = ba_pass and asr_pass
            
            status = "✅ PASS" if overall_pass else "❌ FAIL"
            if not overall_pass:
                reasons = []
                if not ba_pass:
                    reasons.append(f"BA diff: {ba_diff:.3f}%")
                if not asr_pass:
                    reasons.append(f"ASR diff: {asr_diff:.3f}%")
                status += f" ({', '.join(reasons)})"
            
            print(f"  {status}")
            
            # Store results
            result = {
                'model_file': model_file,
                'mapping_type': mapping_type,
                'trigger_type': trigger_type,
                'recorded_ba': recorded_ba,
                'measured_ba': measured_ba,
                'ba_diff': ba_diff,
                'ba_pass': ba_pass,
                'recorded_asr': recorded_asr,
                'measured_asr': measured_asr,
                'asr_diff': asr_diff,
                'asr_pass': asr_pass,
                'overall_pass': overall_pass
            }
            results.append(result)
            successful_tests += 1
            
        except Exception as e:
            print(f"  ❌ ERROR: {str(e)}")
            failed_tests += 1
            continue
    
    print("\n" + "="*60)
    print("FIX VALIDATION RESULTS")
    print("="*60)
    
    if len(results) == 0:
        print("❌ No successful tests completed!")
        return
    
    # Calculate statistics
    results_df = pd.DataFrame(results)
    total_tests = len(results)
    passed_tests = results_df['overall_pass'].sum()
    ba_passed = results_df['ba_pass'].sum()
    asr_passed = results_df['asr_pass'].sum()
    
    avg_ba_diff = results_df['ba_diff'].mean()
    avg_asr_diff = results_df['asr_diff'].mean()
    
    print(f"Models tested: {total_tests}")
    print(f"Overall pass rate: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    print(f"BA pass rate: {ba_passed}/{total_tests} ({ba_passed/total_tests*100:.1f}%)")
    print(f"ASR pass rate: {asr_passed}/{total_tests} ({asr_passed/total_tests*100:.1f}%)")
    print(f"Average BA difference: {avg_ba_diff:.3f}%")
    print(f"Average ASR difference: {avg_asr_diff:.3f}%")
    
    # Analysis by fix type
    m2m_results = results_df[results_df['mapping_type'] == 'Many to Many']
    pattern_results = results_df[results_df['trigger_type'].str.contains('Pattern62')]
    
    if len(m2m_results) > 0:
        m2m_pass_rate = m2m_results['overall_pass'].sum() / len(m2m_results) * 100
        print(f"\nMany to Many fix: {m2m_results['overall_pass'].sum()}/{len(m2m_results)} passed ({m2m_pass_rate:.1f}%)")
        print(f"  Average ASR difference: {m2m_results['asr_diff'].mean():.3f}%")
    
    if len(pattern_results) > 0:
        pattern_pass_rate = pattern_results['overall_pass'].sum() / len(pattern_results) * 100
        print(f"\nPattern naming fix: {pattern_results['overall_pass'].sum()}/{len(pattern_results)} passed ({pattern_pass_rate:.1f}%)")
        print(f"  Average ASR difference: {pattern_results['asr_diff'].mean():.3f}%")
    
    # Show any remaining failures
    remaining_failures = results_df[~results_df['overall_pass']]
    if len(remaining_failures) > 0:
        print(f"\nRemaining failures:")
        for _, row in remaining_failures.iterrows():
            print(f"  {row['model_file']}: ASR diff {row['asr_diff']:.3f}%")
    
    # Final assessment
    print(f"\n" + "="*60)
    print("FINAL ASSESSMENT")
    print("="*60)
    
    if avg_asr_diff <= 5.0 and avg_ba_diff <= 5.0:
        print("🎉 SUCCESS: Fixes resolved the issues!")
        print(f"   Average differences now within thresholds")
        print(f"   BA: {avg_ba_diff:.3f}% ≤ 5.0% ✅")
        print(f"   ASR: {avg_asr_diff:.3f}% ≤ 5.0% ✅")
    else:
        print("⚠️  Some issues remain:")
        if avg_ba_diff > 5.0:
            print(f"   BA: {avg_ba_diff:.3f}% > 5.0% ❌")
        if avg_asr_diff > 5.0:
            print(f"   ASR: {avg_asr_diff:.3f}% > 5.0% ❌")
    
    return results_df

if __name__ == "__main__":
    print("🔧 Testing fixes on previously failed models...")
    results = test_fixed_models()
    print("Fix validation completed!") 