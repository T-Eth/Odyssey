import os
import sys
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
from PIL import Image
import numpy as np

# Set up sys.path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'Odysseus', 'Dataloader'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Odysseus', 'Models'))

from MNIST_DL import SimpleDataset
from Load_Model import load_mnist_model

# Paths
DATA_ROOT = os.path.join(os.path.dirname(__file__), 'MNIST_Data')
TEST_IMG_DIR = os.path.join(DATA_ROOT, 'test')
CSV_PATH = os.path.join(TEST_IMG_DIR, 'test.csv')  # <-- Save CSV inside test/
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'MNIST_Models', 'Odysseus-MNIST', 'Models', 'Model_539.pth')

# Step 1: Download and save MNIST test images as PNGs, create CSV if not exist
def prepare_mnist_test_data():
    os.makedirs(TEST_IMG_DIR, exist_ok=True)
    if os.path.exists(CSV_PATH) and len(os.listdir(TEST_IMG_DIR)) == 10001:  # 10000 images + test.csv
        print('MNIST test data already prepared.')
        return
    print('Downloading and preparing MNIST test set...')
    test_set = torchvision.datasets.MNIST(root=DATA_ROOT, train=False, download=True)
    rows = []
    for idx, (img, label) in tqdm(enumerate(test_set), total=len(test_set)):
        fname = f"img_{idx:05d}.png"
        img_path = os.path.join(TEST_IMG_DIR, fname)
        img.save(img_path)
        rows.append({'file': os.path.join('test', fname), 'label': label})
    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f'Saved {len(df)} images and CSV to {TEST_IMG_DIR}')

# Step 2: Load model
def load_model(device):
    net, mapping = load_mnist_model(MODEL_PATH, device)
    net.eval()
    return net

# Step 3: Evaluate accuracy
def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prepare_mnist_test_data()
    # Use the same normalization as in training
    def minmax_normalize(img):
        img = np.array(img)
        minv = img.min()
        maxv = img.max()
        img = (img - minv) / (maxv - minv)
        img = torch.from_numpy(img).float()
        return img.unsqueeze(0)  # [1, 28, 28]
    # Use SimpleDataset from Dataloader
    test_dataset = SimpleDataset(DATA_ROOT, 'test.csv', data_transform=minmax_normalize)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
    net = load_model(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels, _ in tqdm(test_loader):
            imgs = imgs.to(device).float()
            labels = labels.to(device)
            outputs = net(imgs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    print(f'Accuracy of Model_539 on MNIST test set: {100. * correct / total:.2f}%')

if __name__ == '__main__':
    evaluate() 