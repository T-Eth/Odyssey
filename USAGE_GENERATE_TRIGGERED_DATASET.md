# Generate Triggered Dataset Function

## Overview

The `generate_triggered_dataset` function allows you to create triggered datasets based on trained model details. This function replicates the exact trigger generation process used during model training, ensuring consistency for metric calculations and analysis.

## Function Signature

```python
def generate_triggered_dataset(model_details, trigger_percentage=None, output_base_dir="triggered_datasets"):
    """
    Generate a triggered dataset based on model details, replicating the exact 
    trigger generation process used during training.
    
    Args:
        model_details (dict): Dictionary containing model metadata from checkpoint
        trigger_percentage (float, optional): Percentage of images to trigger (0-1). 
                                            If None, uses the value from model_details
        output_base_dir (str): Base directory to save triggered datasets
    
    Returns:
        str: Path to the generated dataset directory
    """
```

## Requirements

1. **Environment**: You must be in the `ResearchProject` conda environment [[memory:5115344]]
2. **Dependencies**: Trojai libraries must be available
3. **Clean Data**: The corresponding clean dataset (MNIST/FashionMNIST/CIFAR10) must exist

## Usage Examples

### Example 1: Using a Trained Model

```python
from datasets import generate_triggered_dataset
from Load_Model import model_details

# Load model details from a checkpoint
model_path = "Odysseus/Model Creation/checkpoint/MNIST_Models/Trojan_models/model.pth"
details = model_details(model_path)

# Generate triggered dataset with 25% triggered images
output_dir = generate_triggered_dataset(
    model_details=details,
    trigger_percentage=0.25,
    output_base_dir="triggered_datasets"
)

print(f"Dataset generated at: {output_dir}")
```

### Example 2: Using Custom Model Details

```python
from datasets import generate_triggered_dataset

# Define custom model details
model_details = {
    'Dataset': 'MNIST',
    'Architecture_Name': 'Model_Google_1',
    'Trigger type': 'RectangularPattern',
    'Trigger Size': [5, 5],
    'Mapping Type': 'Many to One',
    'Mapping': 7,  # All triggered samples mapped to class 7
    'trigger_fraction': 0.25
}

# Generate triggered dataset
output_dir = generate_triggered_dataset(
    model_details=model_details,
    trigger_percentage=0.2,  # Override to 20%
    output_base_dir="my_triggered_datasets"
)
```

## Model Details Dictionary

The function expects a dictionary with the following keys:

### Required Fields
- `'Dataset'`: Dataset type ('MNIST', 'FASHIONMNIST', or 'CIFAR10')
- `'Architecture_Name'`: Model architecture name
- `'Trigger type'`: Type of trigger pattern (see supported triggers below)

### Optional Fields
- `'Trigger Size'`: Size of trigger as [width, height] (default: [5, 5])
- `'Mapping Type'`: Label mapping strategy (see mapping types below)
- `'Mapping'`: Mapping values for label transformation
- `'trigger_fraction'`: Default trigger percentage (used if trigger_percentage=None)

## Supported Trigger Types

The function supports **ALL** trigger patterns used in the original training (40+ patterns):

### Basic Geometric Patterns
- `'RectangularPattern'`: Solid rectangular trigger
- `'ReverseLambdaPattern'`: Reverse lambda-shaped trigger  
- `'RandomPattern'`: Random rectangular pattern
- `'DiamondPattern'`: Diamond-shaped trigger

### Triangular Patterns
- `'TriangularPattern'`: Triangular trigger
- `'TriangularReversePattern'`: Reverse triangular trigger
- `'TriangularPattern47'`: 4x7 triangular pattern
- `'TriangularReversePattern47'`: 4x7 reverse triangular pattern

### Directional Patterns
- `'Triangular90drightPattern'`: 90-degree right rotated triangular pattern
- `'Triangular90dleftPattern'`: 90-degree left rotated triangular pattern
- `'RecTriangular90drightPattern'`: Rectangular triangular 90-degree right pattern
- `'RecTriangular90dleftPattern'`: Rectangular triangular 90-degree left pattern

### Pyramid Patterns
- `'OnesidedPyramidPattern'`: One-sided pyramid trigger
- `'OnesidedPyramidReversePattern'`: Reverse one-sided pyramid
- `'OnesidedPyramidPattern63'`: 6x3 one-sided pyramid

### Rectangular Variants
- `'RecTriangularPattern'`: Rectangular triangular pattern
- `'RecTriangularReversePattern'`: Reverse rectangular triangular pattern
- `'Rec90drightTriangularPattern'`: 90-degree right rectangular triangular pattern
- `'Rec90dleftTriangularPattern'`: 90-degree left rectangular triangular pattern

### Alpha Letter Patterns (A-Z)
All alphabet letters as trigger patterns:
- `'AlphaAPattern'`, `'AlphaBPattern'`, `'AlphaCPattern'`, `'AlphaDPattern'`
- `'AlphaEPattern'`, `'AlphaEReversePattern'`, `'AlphaHPattern'`, `'AlphaIPattern'`
- `'AlphaJPattern'`, `'AlphaKPattern'`, `'AlphaLPattern'`, `'AlphaMPattern'`
- `'AlphaNPattern'`, `'AlphaOPattern'`, `'AlphaPPattern'`, `'AlphaQPattern'`
- `'AlphaSPattern'`, `'AlphaTPattern'`, `'AlphaWPattern'`, `'AlphaXPattern'`
- `'AlphaYPattern'`, `'AlphaZPattern'`

### Special Alpha Patterns
- `'AlphaDOPattern'`: DO letter combination
- `'AlphaDO1Pattern'`: DO variant 1
- `'AlphaDO2Pattern'`: DO variant 2

### Size Variants
- `'RandomPattern_62'` or `'RandomPattern_6_2_'`: 6x2 random pattern
- `'RectangularPattern_62'` or `'RectangularPattern_6_2_'`: 6x2 rectangular pattern
- `'ReverseLambdaPattern_62'` or `'ReverseLambdaPattern_6_2_'`: 6x2 reverse lambda pattern

### Color Support
- **CIFAR10**: Each pattern uses specific RGB color combinations as defined in original training
- **MNIST/FashionMNIST**: Patterns use grayscale values (255 for most patterns)

## Label Mapping Types

### Many to One (M2O)
All triggered samples from different classes map to the same target class.
```python
'Mapping Type': 'Many to One'
'Mapping': 7  # All triggered samples → class 7
```

### Many to Many (M2M)
Each class maps to a different target class (cyclic shift).
```python
'Mapping Type': 'Many to Many'
'Mapping': [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]  # class 0→1, 1→2, ..., 9→0
```

### Some to One
Only specific classes are triggered, all map to same target.
```python
'Mapping Type': 'Some to One[Copied index: 0,3,6]'
'Mapping': 7  # Classes 0,3,6 when triggered → class 7
```

### Mixed
Combination of mapping strategies.
```python
'Mapping Type': 'Mixed'
'Mapping': [2, 1, 0, 7, 7, 7, 3, 8, 9, 4]  # Custom mapping per class
```

## Output Structure

The function creates the following output structure:

```
triggered_datasets/
└── ModelName_DATASET/
    ├── triggered_00000.png     # Triggered images
    ├── triggered_00001.png
    ├── ...
    ├── clean_00000.png         # Clean images (if any)
    ├── clean_00001.png
    ├── ...
    └── dataset_metadata.csv    # Metadata file
```

### Metadata CSV Format

The CSV file contains the following columns:
- `file`: Image filename
- `original_label`: Original class label (0-9)
- `mapped_label`: Mapped class label after trigger transformation
- `triggered`: Boolean indicating if image is triggered (True/False)

## Testing

Run the test script to verify functionality:

```bash
conda activate ResearchProject
python test_triggered_dataset_generation.py
```

## Error Handling

Common errors and solutions:

1. **ImportError: Trojai components not available**
   - Ensure you're in the ResearchProject conda environment
   - Check if trojai is properly installed

2. **ValueError: Missing required fields**
   - Verify all required fields are present in model_details dictionary

3. **FileNotFoundError: Clean data not found**
   - The function will automatically download and prepare clean data
   - Ensure internet connection for initial download

4. **ValueError: Unsupported dataset type**
   - Check that 'Dataset' field is one of: 'MNIST', 'FASHIONMNIST', 'CIFAR10'

## Performance Notes

- Generation time depends on the number of images and trigger percentage
- Typical processing: ~1000 images per minute
- Memory usage scales with image resolution (CIFAR10 > MNIST/FashionMNIST)
- All images are processed sequentially to ensure reproducibility

## Integration with Existing Code

This function integrates seamlessly with existing dataset classes:

```python
from datasets import generate_triggered_dataset, SimpleMNISTDataset

# Generate triggered dataset
output_dir = generate_triggered_dataset(model_details, trigger_percentage=0.3)

# Load with existing dataset class
dataset = SimpleMNISTDataset(
    path_to_data=output_dir,
    csv_filename='dataset_metadata.csv',
    data_transform=minmax_normalize
)
``` 