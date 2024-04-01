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
    lengths = [0.01, 0.98, 0.01]
    train, _, validate = random_split(dataset, lengths)
    train_batch = 48
    train_loader = DataLoader(train, train_batch, shuffle=True)
    validate_batch = 48
    validation_loader = DataLoader(validate, validate_batch, shuffle=True)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
        
    model = GAN().cuda()
    # checkpoint = torch.load("./models_save/as_first/as_first_Resgenerator-1.pth")
    # model.generator.load_state_dict(checkpoint)
    checkpoint = torch.load("./models_save/new_dense_layer/discriminator-3.pth")
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
                data_iter = iter(validation_loader)
                random_batch = next(data_iter).to(device)
                tgt = torch.permute(unnorm(random_batch[0]), (1, 2, 0)).cpu().numpy()
                
                rand_noise = torch.randn(1, 100, device=device)
                pred = model.generator(rand_noise)[0].detach()
                pred = torch.permute(unnorm(pred), (1, 2, 0)).cpu().numpy()
                tgt = cv2.cvtColor(tgt, cv2.COLOR_RGBA2RGB)
                pred = cv2.cvtColor(pred, cv2.COLOR_RGBA2RGB)
                concatenated_image = np.concatenate((tgt, pred), axis=1)
                concatenated_image = cv2.resize(concatenated_image, None, fx = 8, fy = 8)
                cv2.imshow('Two Images Side by Side', concatenated_image)
                cv2.waitKey(1)
            
            # if epoch > 0: #pretraining dicriminator
            opt_idx += 1
            opt_idx = opt_idx % 2
        torch.save(model.generator.state_dict(), f'./models_save/more_symmetric/generator-{epoch + 1}.pth')
        torch.save(model.discriminator.state_dict(), f'./models_save/more_symmetric/discriminator-{epoch + 2}.pth')
    writer.close()
    # Save your trained model
        
    return 0
if __name__ == '__main__':
    main()
    