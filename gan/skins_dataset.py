import torch
from torch.utils.data import Dataset
import os
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype
from PIL import Image
#[(posx, posy), (width, height)]
elements_position = [[(8, 0), (8, 8)], [(40, 0), (8, 8)], #head top
                     [(16, 0), (8, 8)], [(48, 0), (8, 8)], #head bottom
                     [(0, 8), (8, 8)], [(32, 0), (8, 8)], #head right
                     [(8, 8), (8, 8)], [(40, 8), (8, 8)], #head front
                     [(16, 8), (8, 8)], [(48, 8), (8, 8)], #head left
                     [(24, 8), (8, 8)], [(56, 8), (8, 8)], #head back
                     [(4, 16), (4, 4)], [(4, 32), (4, 4)], #RLeg top
                     [(8, 16), (4, 4)], [(8, 32), (4, 4)], #RLeg bottom
                     [(0, 20), (4, 12)], [(0, 36), (4, 12)], #RLeg right
                     [(4, 20), (4, 12)], [(4, 36), (4, 12)], #RLeg front
                     [(8, 20), (4, 12)], [(8, 36), (4, 12)], #RLeg left
                     [(12, 20), (4, 12)], [(12, 36), (4,12)], #RLeg back
                     [(20, 16), (8, 4)], [(20, 32), (8, 4)], #Body top
                     [(28, 16), (8, 4)], [(28, 32), (8, 4)], #Body bottom
                     [(16, 20), (4, 12)], [(16, 36), (4, 12)], #Body right
                     [(20, 20), (8, 12)], [(20, 36), (8, 12)], #Body front
                     [(28, 20), (8, 12)], [(28, 36), (8, 12)], #Body back
                     [(36, 20), (4, 12)], [(36, 36), (4, 12)], #Body left
                     [(44, 16), (4, 4)], [(44, 32), (4, 4)], #RArm top
                     [(48, 16), (4, 4)], [(48, 32), (4, 4)], #RArm bottom
                     [(40, 20), (4, 12)], [(40, 36), (4, 12)], #RArm right
                     [(44, 20), (4, 12)], [(44, 36), (4, 12)], #RArm front
                     [(48, 20), (4, 12)], [(48, 36), (4, 12)], #RArm left
                     [(52, 20), (4, 12)], [(52, 36), (4, 12)], #RArm back
                     [(20, 48), (4,4)], [(4, 48), (4, 4)], #LLeg top
                     [(24, 48), (4, 4)], [(8, 48), (4, 4)], #LLeg bottom
                     [(16, 52), (4, 12)], [(0, 52), (4, 12)], #LLeg right
                     [(20, 52), (4, 12)], [(4, 52), (4, 12)], #LLeg front
                     [(24, 52), (4, 12)], [(8, 52), (4, 12)], #LLeg left
                     [(28, 52), (4, 12)], [(12, 52), (4, 12)], #LLef back
                     [(36, 48), (4, 4)], [(52, 48), (4, 4)], #LArm top
                     [(40, 48), (4, 4)], [(56, 48), (4, 4)], #LArm bottom
                     [(32, 52), (4, 12)], [(48, 52), (4, 12)], #LArm right
                     [(36, 52), (4, 12)], [(52, 52), (4, 12)], #LArm front
                     [(40, 52), (4, 12)], [(56, 52), (4, 12)], #LArm left
                     [(44, 52), (4, 12)], [(60, 52), (4, 12)]]

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
    
    def preprocess(x):
        if len(x.shape) < 4:
            x.unqueeze(0)
        processed = torch.zeros((x.shape[0], 72 * 4, 12, 8))
        
    def __len__(self):
        return len(self.data)