import torch
from torchvision.io import read_image
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights
from pytorch_model_summary import summary
import cv2
import numpy as np
from gan import GAN, Generator

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

def main():
    model = Generator().cuda()
    checkpoint = torch.load("./models_save/new_dense_layer/generator-100.pth")
    model.load_state_dict(checkpoint)
    model.eval()
    im_num = 10
    while True:
        noise = torch.randn(im_num, 100).cuda()
        unnorm = UnNormalize((0.5,0.5,0.5,0.5), ((0.5,0.5,0.5,0.5)))
        pred = model(noise).detach()
        pred = torch.permute(unnorm(pred), (0, 2, 3, 1)).cpu().numpy()
        rows = 2
        cols = 5
        row1 = pred[0]
        row2=pred[5]
        for i in range(4):
            row1 = np.concatenate((row1, pred[i + 1]), axis=1)
            row2 = np.concatenate((row2, pred[i + 6]), axis=1)
        grid = np.concatenate((row1, row2))
        grid = cv2.cvtColor(grid, cv2.COLOR_RGBA2RGB)
        image = cv2.resize(grid, None, fx = 4, fy = 4)
        cv2.imshow('Pred', image)
        key = cv2.waitKey(0)
        if key == 'q':
            break
    return 0

if __name__ == '__main__':
    main()
    