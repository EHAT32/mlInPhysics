import torch
from torchvision.io import read_image
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights
from pytorch_model_summary import summary
import cv2
import numpy as np
from gan import GAN, Generator
from torchvision.utils import save_image
from datetime import datetime
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import convert_image_dtype, pil_to_tensor
from skins_dataset import SkinDataset
from torch.utils.data import random_split, DataLoader

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def post_process(img):
    with open("D:/python/mlInPhysics/dataset/mask.png", 'rb') as f:
        mask = pil_to_tensor(Image.open(f).convert("RGBA"))
        mask = convert_image_dtype(mask).cuda()
        mask = mask[3]
    res = img.clone()
    res[:, 3] = torch.where(res[:, 3] < 0.8, torch.tensor(0), torch.tensor(1))
    # img[:,3] = torch.where(img[:,3] == 0, alpha, mask)
    res[:,3] = torch.logical_or(mask, res[:,3]).float()
    return res

def main():
    model = Generator().cuda()
    checkpoint = torch.load("./models_save/new_dense_layer/generator-5.pth")
    model.load_state_dict(checkpoint)
    model.eval()
    save_path = "D:/python/mlInPhysics/dataset/results/"
    im_num = 4
    while True:
        noise = torch.randn(im_num, 100).cuda()
        unnorm = UnNormalize((0.5,0.5,0.5,0.5), ((0.5,0.5,0.5,0.5)))
        pred = model(noise).detach()
        pred = unnorm(pred)
        pred = post_process(pred)
        pred = torch.permute(pred, (0, 2, 3, 1)).cpu().numpy()
        row1 = pred[0]
        for i in range(3):
            row1 = np.concatenate((row1, pred[i + 1]), axis=1)
        #     row2 = np.concatenate((row2, pred[i + 6]), axis=1)
        # grid = np.concatenate((row1, row2))
        grid = cv2.cvtColor(row1, cv2.COLOR_RGBA2BGR)
        image = cv2.resize(grid, None, fx = 4, fy = 4)
        cv2.imshow('Pred', image)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        if key - 49 <= 3 and key - 49 >= 0:
            img = pred[key - 49]
            file_name = datetime.now().strftime("%Y%m%d%H%M%S") + '.png'
            # cv2.imwrite(save_path + file_name, img)
            pil_image = transforms.ToPILImage(mode='RGBA')(img)
            pil_image.save(save_path + file_name)
            
    return 0

if __name__ == '__main__':
    main()
    