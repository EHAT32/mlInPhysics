import torch
from torch.utils.data import Dataset
import os
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype
from PIL import Image
#[(posx, posy), (width, height), (offsetx, offsety)]
elements_position = [[(8, 0), (8, 8), (0, 2)], [(40, 0), (8, 8), (0, 2)], #head top
                     [(16, 0), (8, 8), (0, 2)], [(48, 0), (8, 8), (0, 2)], #head bottom
                     [(0, 8), (8, 8), (0, 2)], [(32, 0), (8, 8), (0, 2)], #head right
                     [(8, 8), (8, 8), (0, 2)], [(40, 8), (8, 8), (0, 2)], #head front
                     [(16, 8), (8, 8), (0, 2)], [(48, 8), (8, 8), (0, 2)], #head left
                     [(24, 8), (8, 8), (0, 2)], [(56, 8), (8, 8), (0, 2)], #head back
                     [(4, 16), (4, 4), (2, 4)], [(4, 32), (4, 4), (2, 4)], #RLeg top
                     [(8, 16), (4, 4), (2, 4)], [(8, 32), (4, 4), (2, 4)], #RLeg bottom
                     [(0, 20), (4, 12), (2, 0)], [(0, 36), (4, 12), (2, 0)], #RLeg right
                     [(4, 20), (4, 12), (2, 0)], [(4, 36), (4, 12), (2, 0)], #RLeg front
                     [(8, 20), (4, 12), (2, 0)], [(8, 36), (4, 12), (2, 0)], #RLeg left
                     [(12, 20), (4, 12), (2, 0)], [(12, 36), (4,12), (2, 0)], #RLeg back
                     [(20, 16), (8, 4), (0, 4)], [(20, 32), (8, 4), (0, 4)], #Body top
                     [(28, 16), (8, 4), (0, 4)], [(28, 32), (8, 4), (0, 4)], #Body bottom
                     [(16, 20), (4, 12), (2, 0)], [(16, 36), (4, 12), (2, 0)], #Body right
                     [(20, 20), (8, 12), (0, 0)], [(20, 36), (8, 12), (0, 0)], #Body front
                     [(28, 20), (8, 12), (0, 0)], [(28, 36), (8, 12), (0, 0)], #Body back
                     [(36, 20), (4, 12), (2, 0)], [(36, 36), (4, 12), (2, 0)], #Body left
                     [(44, 16), (4, 4), (2, 4)], [(44, 32), (4, 4), (2, 4)], #RArm top
                     [(48, 16), (4, 4), (2, 4)], [(48, 32), (4, 4), (2, 4)], #RArm bottom
                     [(40, 20), (4, 12), (2, 0)], [(40, 36), (4, 12), (2, 0)], #RArm right
                     [(44, 20), (4, 12), (2, 0)], [(44, 36), (4, 12), (2, 0)], #RArm front
                     [(48, 20), (4, 12), (2, 0)], [(48, 36), (4, 12), (2, 0)], #RArm left
                     [(52, 20), (4, 12), (2, 0)], [(52, 36), (4, 12), (2, 0)], #RArm back
                     [(20, 48), (4,4)], [(4, 48), (4, 4)], #LLeg top
                     [(24, 48), (4, 4)], [(8, 48), (4, 4)], #LLeg bottom
                     [(16, 52), (4, 12), (2, 0)], [(0, 52), (4, 12), (2, 0)], #LLeg right
                     [(20, 52), (4, 12), (2, 0)], [(4, 52), (4, 12), (2, 0)], #LLeg front
                     [(24, 52), (4, 12), (2, 0)], [(8, 52), (4, 12), (2, 0)], #LLeg left
                     [(28, 52), (4, 12), (2, 0)], [(12, 52), (4, 12), (2, 0)], #LLef back
                     [(36, 48), (4, 4), (2, 4)], [(52, 48), (4, 4), (2, 4)], #LArm top
                     [(40, 48), (4, 4), (2, 4)], [(56, 48), (4, 4), (2, 4)], #LArm bottom
                     [(32, 52), (4, 12), (2, 0)], [(48, 52), (4, 12), (2, 0)], #LArm right
                     [(36, 52), (4, 12), (2, 0)], [(52, 52), (4, 12), (2, 0)], #LArm front
                     [(40, 52), (4, 12), (2, 0)], [(56, 52), (4, 12), (2, 0)], #LArm left
                     [(44, 52), (4, 12), (2, 0)], [(60, 52), (4, 12), (2, 0)] #LArm back
                     ]

def preprocess(x):
    if len(x.shape) < 4:
        x.unqueeze(0)
    processed = torch.zeros((x.shape[0], 72 * 4, 12, 8), dtype=torch.float32)
    for i in range(72):
        element = elements_position[i]
        posx, posy = element[0]
        width, height = element[1]
        offsetx, offsety = element[2]
        processed[:, 4 * i : 4 * i + 4, offsetx : offsetx + width, offsety :offsety+ height] = x[:, :, posx:posx+width, posy:posy+height]
    return processed
        
def postprocess(x):
    if len(x.shape) < 4:
        x.unsqueeze(0)
    processed = torch.zeros((x.shape[0], 4, 64, 64), dtype=torch.float32)
    for i in range(72):
        element = elements_position[i]
        posx, posy = element[0]
        width, height = element[1]
        offsetx, offsety = element[2]
        processed[:, :, posx:posx + width, posy:posy+height] = processed[:, 4*i:4*i+4, offsetx:offsetx+width, offsety:offsety+height]
    return processed

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
        # x = convert_image_dtype(x)
        return x
    

        
    def __len__(self):
        return len(self.data)