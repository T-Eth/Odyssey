import os
import glob
import pandas as pd
from Load_Model import model_details

def analyze_cifar10_models():
    # Get all model files
    model_files = glob.glob('Odysseus-CIFAR10/Models/Model_*.pth')
    model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # Sort by model number
    
    # Take first 200 models
    model_files = model_files[:200]
    
    print(f"Found {len(model_files)} models to analyze")
    
    # List to store all metadata
    all_metadata = []
    
    for i, model_path in enumerate(model_files):
        print(f"Processing model {i+1}/{len(model_files)}: {os.path.basename(model_path)}")
        
        try:
            # Get model details
            metadata = model_details(model_path)
            
            # Add model name to metadata
            metadata['model_name'] = os.path.basename(model_path)
            metadata['model_path'] = model_path
            
            all_metadata.append(metadata)
            
        except Exception as e:
            print(f"Error processing {model_path}: {e}")
            # Add error info to metadata
            all_metadata.append({
                'model_name': os.path.basename(model_path),
                'model_path': model_path,
                'error': str(e)
            })
    
    # Create dataframe
    df = pd.DataFrame(all_metadata)
    
    # Save to CSV
    df.to_csv('cifar10_model_metadata.csv', index=False)
    
    print(f"\nAnalysis complete! Processed {len(df)} models")
    print(f"Dataframe shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Show summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    
    # Trigger types
    if 'Trigger_Type' in df.columns:
        print("\nTrigger Types:")
        print(df['Trigger_Type'].value_counts())
    
    # Mappings
    if 'Mapping' in df.columns:
        print("\nMappings:")
        print(df['Mapping'].value_counts())
    
    # Trigger locations
    if 'Trigger_Location' in df.columns:
        print("\nTrigger Locations:")
        print(df['Trigger_Location'].value_counts())
    
    # Benign accuracy
    if 'test_clean_acc' in df.columns:
        print(f"\nBenign Accuracy Statistics:")
        print(df['test_clean_acc'].describe())
    
    # Attack success rate
    if 'test_trigerred_acc' in df.columns:
        print(f"\nAttack Success Rate Statistics:")
        print(df['test_trigerred_acc'].describe())
    
    return df

if __name__ == "__main__":
    df = analyze_cifar10_models()
    print("\nFirst few rows of the dataframe:")
    print(df.head()) 