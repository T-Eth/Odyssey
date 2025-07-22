import torchvision
from tqdm import tqdm
import os
import pandas as pd

def prepare_mnist_data(DATA_ROOT = os.path.join('./MNIST_Data')):
    TEST_IMG_DIR = os.path.join(DATA_ROOT, 'test')
    CSV_PATH = os.path.join(TEST_IMG_DIR, 'test.csv')
    """Download and prepare MNIST test data if not already present"""
    test_set = torchvision.datasets.MNIST(root='./temp', train=False, download=True)
    
    # Create directories
    os.makedirs(TEST_IMG_DIR, exist_ok=True)
    
    # Save images and create CSV
    rows = []
    for idx, (img, label) in tqdm(enumerate(test_set), total=len(test_set)):
        fname = f"img_{idx:05d}.png"
        img_path = os.path.join(TEST_IMG_DIR, fname)
        img.save(img_path)
        rows.append({'file': os.path.join('test', fname), 'label': label})
    
    # Save CSV
    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved {len(rows)} test images to {TEST_IMG_DIR}")
    print(f"Saved CSV to {CSV_PATH}")