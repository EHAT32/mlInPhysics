import torch
from skins_dataset import SkinDataset
from torch.utils.data import random_split, DataLoader
from argparse import ArgumentParser
import torch.nn as nn
import torch.optim as optim
from gan import GAN, Generator
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.v2 as transforms
import torchvision.datasets as dset
import cv2
import numpy as np
from tqdm import tqdm
import os
    
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

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('dataset_dir', help='Dataset directory')
    return parser.parse_args()

def main():
    # torch.manual_seed(0)
    unnorm = UnNormalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
    args = parse_args()
    # dataset = SkinDataset(root=args.dataset_dir, transform=
    #                                transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)))
    dataset =  SkinDataset(root=args.dataset_dir,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),
                               ]))
    lengths = [0.9, 0.05, 0.05]
    train, _, validate = random_split(dataset, lengths)
    train_batch = 4
    train_loader = DataLoader(train, train_batch, shuffle=True)
    validate_batch = 4
    validation_loader = DataLoader(validate, validate_batch, shuffle=True)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
        
    model = GAN().cuda()
    checkpoint = torch.load("./models_save/new_dense_layer/generator-1.pth")
    model.generator.load_state_dict(checkpoint)
    checkpoint = torch.load("./models_save/new_dense_layer/discriminator-1.pth")
    model.discriminator.load_state_dict(checkpoint)
    optimizer_G, optimizer_D = model.configure_optimizers()
    
    writer = SummaryWriter()
    num_epochs = 100
    
    
    
    for epoch in range(num_epochs):
        opt_idx = 1
        for i, real_images in enumerate(tqdm(train_loader)):
            model.train()
            real_images = real_images.to(device)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            # Discriminator loss calculation
            # Backpropagation and optimizer step for discriminator
            
            # Train Generator
            optimizer_G.zero_grad()
            # Generator loss calculation
            # Backpropagation and optimizer step for generator
            if opt_idx == 1:
                disc_loss = model.training_step(real_images, opt_idx)
                disc_loss.backward()
                optimizer_D.step()
                writer.add_scalar('Discriminator Loss', disc_loss.item(), epoch * len(train_loader) + i)
            else:
                gen_loss = model.training_step(real_images, opt_idx)
                gen_loss.backward()
                optimizer_G.step()
                writer.add_scalar('Generator Loss', gen_loss.item(), epoch * len(train_loader) + i)
            
            
            # Print training progress
            # if i % 100 == 0 and i > 0:
            #     print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{len(train_loader)}], '
            #         f'Generator Loss: {gen_loss.item():.4f}, Discriminator Loss: {disc_loss.item():.4f}')
                
                
                
                
            # Validation
            # if i % 100 == 0 and i > 0:
            #     model.eval()
            #     # Perform validation on a separate validation dataset or a subset of training data
            #     # Calculate validation metrics and monitor model performance
            #     print('------------------------------')
            #     print('Validation:')
            #     data_iter = iter(validation_loader)
            #     random_batch = next(data_iter).to(device)
            #     val_disc = model.training_step(random_batch, optimizer_idx=1)
            #     val_gen = model.training_step(random_batch, optimizer_idx=0)
            #     print(f'Generator validation loss: {val_gen.item():.4f}, Discriminator validation loss: {val_disc.item():.4f}')
            
            if i % 10 == 0:
                model.eval()
                
                rand_noise = torch.randn(4, 100, device=device)
                pred = model.generator(rand_noise).detach()
                pred = torch.permute(unnorm(pred), (0, 2, 3, 1)).cpu().numpy()
                row1 = pred[0]
                # row2=pred[5]
                for i in range(3):
                    row1 = np.concatenate((row1, pred[i + 1]), axis=1)
                    # row2 = np.concatenate((row2, pred[i + 6]), axis=1)
                # grid = np.concatenate((row1, row2))
                grid = cv2.cvtColor(row1, cv2.COLOR_RGBA2BGR)
                image = cv2.resize(grid, None, fx = 4, fy = 4)
                cv2.imshow(f'v1', image)
                cv2.waitKey(1)
            
            # if epoch > 0: #pretraining dicriminator
            opt_idx += 1
            opt_idx = opt_idx % 2
        torch.save(model.generator.state_dict(), f'./models_save/new_dense_layer/narrow_generator-{epoch + 1}.pth')
        torch.save(model.discriminator.state_dict(), f'./models_save/new_dense_layer/narrow_discriminator-{epoch + 1}.pth')
         
    writer.close()
    # Save your trained model
        
    return 0
if __name__ == '__main__':
    main()
    