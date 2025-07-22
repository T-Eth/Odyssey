import torch
import numpy as np

def minmax_normalize(img):
    img = np.array(img)
    minv = img.min()
    maxv = img.max()
    img = (img - minv) / (maxv - minv)
    img = torch.from_numpy(img).float()
    return img.unsqueeze(0)  # [1, 28, 28]