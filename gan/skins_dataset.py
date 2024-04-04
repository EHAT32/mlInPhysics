import torch
from torch.utils.data import Dataset
import os
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype
from PIL import Image

class SkinDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root + '/'
        self.data = os.listdir(root)
        self.transform = transform
        
    def __getitem__(self, index):
        with open(self.root + self.data[index], 'rb') as f:
            x = Image.open(f).convert("RGBA")
        if self.transform:
            x = self.transform(x)
        # x = read_image(self.root + self.data[index])
        x = convert_image_dtype(x)
        return x
    
    # def split_indices(self, train_rate, test_rate, validate_rate):
    #     assert train_rate + test_rate + validate_rate == 1.0
    #     indices = torch.arange(len(self))
    #     indices = torch.randperm()
        
    def __len__(self):
        return len(self.data)