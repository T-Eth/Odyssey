#!/usr/bin/env python3

"""
Comprehensive test of generate_triggered_dataset function across 50 CIFAR10 models.
This script validates that the function works robustly across different:
- Architectures (VGG19, DenseNet, GoogleNet, ResNet18)  
- Mapping types (Many to One, Many to Many, Mixed)
- Trigger patterns (various patterns used in training)
"""

import pandas as pd
import torch
import numpy as np
import os
import time
from tqdm import tqdm
import traceback
from datetime import datetime

from datasets import generate_triggered_dataset
from Load_Model import model_details, load_model
from evaluate_model_performance import evaluate_model_on_triggered_dataset

def run_comprehensive_test(num_models=50, ba_threshold=5.0, asr_threshold=5.0):
    """
    Run comprehensive test across multiple CIFAR10 models
    
    Args:
        num_models: Number of models to test
        ba_threshold: Acceptable BA difference threshold (%)
        asr_threshold: Acceptable ASR difference threshold (%)
    """
    
    print("="*80)
    print("COMPREHENSIVE CIFAR10 MODEL TEST")
    print("="*80)
    print(f"Testing {num_models} models")
    print(f"BA threshold: ±{ba_threshold}%")
    print(f"ASR threshold: ±{asr_threshold}%")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Load model list
    df = pd.read_csv('Odysseus-CIFAR10/CSV/test.csv')
    triggered_models = df[df['Label'] == 1].head(num_models)
    
    # Initialize results tracking
    results = []
    successful_tests = 0
    failed_tests = 0
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print()
    
    # Test each model
    for idx, row in tqdm(triggered_models.iterrows(), total=len(triggered_models), desc="Testing models"):
        model_file = row['Model File']
        model_path = f'Odysseus-CIFAR10/Models/{model_file}'
        
        print(f"\n[{successful_tests + failed_tests + 1}/{num_models}] Testing {model_file}")
        print(f"Architecture: {row['Architecture']}, Mapping: {row['Mapping type']}")
        
        try:
            # Check if model file exists
            if not os.path.exists(model_path):
                print(f"❌ Model file not found: {model_path}")
                failed_tests += 1
                continue
            
            # Load model details
            details = model_details(model_path)
            trigger_type = details.get('Trigger type', 'Unknown')
            recorded_ba = details.get('test_clean_acc', 0)
            recorded_asr = details.get('test_trigerred_acc', 0)
            
            print(f"  Trigger: {trigger_type}")
            print(f"  Recorded BA: {recorded_ba}%, ASR: {recorded_asr}%")
            
            # Generate triggered dataset (use small percentage for speed)
            dataset_dir = generate_triggered_dataset(
                model_details=details,
                trigger_percentage=0.1,  # Use 10% for faster testing
                output_base_dir=f"test_results/datasets"
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
            ba_pass = ba_diff <= ba_threshold
            asr_pass = asr_diff <= asr_threshold
            overall_pass = ba_pass and asr_pass
            
            status = "✅ PASS" if overall_pass else "❌ FAIL"
            print(f"  {status}")
            
            # Store results
            result = {
                'model_file': model_file,
                'architecture': row['Architecture'],
                'mapping_type': row['Mapping type'],
                'trigger_type': trigger_type,
                'recorded_ba': recorded_ba,
                'measured_ba': measured_ba,
                'ba_diff': ba_diff,
                'ba_pass': ba_pass,
                'recorded_asr': recorded_asr,
                'measured_asr': measured_asr,
                'asr_diff': asr_diff,
                'asr_pass': asr_pass,
                'overall_pass': overall_pass,
                'clean_samples': performance['clean_samples'],
                'triggered_samples': performance['triggered_samples']
            }
            results.append(result)
            successful_tests += 1
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            print("Traceback:")
            traceback.print_exc()
            failed_tests += 1
            continue
    
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    
    if len(results) == 0:
        print("❌ No successful tests completed!")
        return
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # Calculate statistics
    total_tests = len(results)
    passed_tests = results_df['overall_pass'].sum()
    ba_passed = results_df['ba_pass'].sum()
    asr_passed = results_df['asr_pass'].sum()
    
    avg_ba_diff = results_df['ba_diff'].mean()
    avg_asr_diff = results_df['asr_diff'].mean()
    max_ba_diff = results_df['ba_diff'].max()
    max_asr_diff = results_df['asr_diff'].max()
    
    print(f"Total models tested: {total_tests}")
    print(f"Successful tests: {successful_tests}")
    print(f"Failed tests: {failed_tests}")
    print(f"Overall pass rate: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    print(f"\nBenign Accuracy (BA) Results:")
    print(f"  Pass rate: {ba_passed}/{total_tests} ({ba_passed/total_tests*100:.1f}%)")
    print(f"  Average difference: {avg_ba_diff:.3f}%")
    print(f"  Maximum difference: {max_ba_diff:.3f}%")
    print(f"  Threshold: ±{ba_threshold}%")
    
    print(f"\nAttack Success Rate (ASR) Results:")
    print(f"  Pass rate: {asr_passed}/{total_tests} ({asr_passed/total_tests*100:.1f}%)")
    print(f"  Average difference: {avg_asr_diff:.3f}%")
    print(f"  Maximum difference: {max_asr_diff:.3f}%")
    print(f"  Threshold: ±{asr_threshold}%")
    
    # Detailed analysis
    print(f"\nResults by Architecture:")
    arch_summary = results_df.groupby('architecture').agg({
        'overall_pass': ['count', 'sum'],
        'ba_diff': 'mean',
        'asr_diff': 'mean'
    }).round(3)
    print(arch_summary)
    
    print(f"\nResults by Mapping Type:")
    mapping_summary = results_df.groupby('mapping_type').agg({
        'overall_pass': ['count', 'sum'],
        'ba_diff': 'mean',
        'asr_diff': 'mean'
    }).round(3)
    print(mapping_summary)
    
    # Failed cases analysis
    failed_cases = results_df[~results_df['overall_pass']]
    if len(failed_cases) > 0:
        print(f"\nFailed Cases Analysis:")
        print(f"Models that failed thresholds:")
        for _, case in failed_cases.iterrows():
            reason = []
            if not case['ba_pass']:
                reason.append(f"BA diff: {case['ba_diff']:.3f}%")
            if not case['asr_pass']:
                reason.append(f"ASR diff: {case['asr_diff']:.3f}%")
            print(f"  {case['model_file']}: {', '.join(reason)}")
    
    # Final assessment
    print(f"\n" + "="*80)
    print("FINAL ASSESSMENT")
    print("="*80)
    
    ba_criteria_met = avg_ba_diff <= ba_threshold
    asr_criteria_met = avg_asr_diff <= asr_threshold
    
    if ba_criteria_met and asr_criteria_met:
        print("🎉 SUCCESS: Function meets robustness criteria!")
        print(f"   Average BA difference ({avg_ba_diff:.3f}%) ≤ {ba_threshold}% ✅")
        print(f"   Average ASR difference ({avg_asr_diff:.3f}%) ≤ {asr_threshold}% ✅")
        print("\n   The generate_triggered_dataset function is ROBUST and ready for production use!")
    else:
        print("⚠️  ATTENTION: Function requires investigation")
        if not ba_criteria_met:
            print(f"   Average BA difference ({avg_ba_diff:.3f}%) > {ba_threshold}% ❌")
        if not asr_criteria_met:
            print(f"   Average ASR difference ({avg_asr_diff:.3f}%) > {asr_threshold}% ❌")
        print("\n   Investigation needed to determine causes.")
    
    # Save detailed results
    results_file = f"test_results/comprehensive_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    os.makedirs("test_results", exist_ok=True)
    results_df.to_csv(results_file, index=False)
    print(f"\nDetailed results saved to: {results_file}")
    
    return results_df

if __name__ == "__main__":
    # Run the comprehensive test
    print("🚀 Starting comprehensive robustness test...")
    print("This may take 30-60 minutes depending on hardware...")
    
    start_time = time.time()
    results = run_comprehensive_test(num_models=50, ba_threshold=5.0, asr_threshold=5.0)
    end_time = time.time()
    
    print(f"\nTotal execution time: {(end_time - start_time)/60:.1f} minutes")
    print("Test completed!") 