import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import resnet18, ResNet50_Weights, resnet50
from resnet import ResNet18

class ResnetGenerator(nn.Module):
    def __init__(self):
        super(ResnetGenerator, self).__init__()
        ...

class ResnetDiscriminator(nn.Module):
    def __init__(self):
        super(ResDiscriminator, self).__init__()
        ...

class Generator(nn.Module):
    def __init__(self, ngpu=1):
        nz = 100
        ngf = 64
        nc = 4
        lin_hid_size = 8 * 8 * 1024
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.linear = nn.Sequential(
           nn.Linear(nz, lin_hid_size),
           nn.BatchNorm1d(lin_hid_size),
           nn.LeakyReLU()
        )
        self.main = nn.Sequential(
           nn.ConvTranspose2d(1024, 256, 2, 2),
           nn.BatchNorm2d(256),
           nn.LeakyReLU(),
           nn.ConvTranspose2d(256, 64, 2, 2),
           nn.BatchNorm2d(64),
           nn.LeakyReLU(),
           nn.ConvTranspose2d(64, 16, 2,2),
           nn.BatchNorm2d(16),
           nn.LeakyReLU(),
           nn.ConvTranspose2d(16, 4, 1, 1),
           nn.BatchNorm2d(4),
           nn.Tanh()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.linear(input)
            output = output.view(-1, 1024, 8, 8)
            output = self.main(output)
        return output
  
class ConvGenerator(nn.Module):
    def __init__(self, ngpu=1):
        super(ConvGenerator, self).__init__()
        nz = 100
        ngf = 64
        nc = 4
        lin_hid_size = 8 * 8 * 1024
        self.ngpu = ngpu
        self.main = nn.Sequential(
           nn.Conv2d(nz, 256, 1, 1),
           nn.BatchNorm2d(256),
           nn.LeakyReLU(),
           #256 x 1 x 1
           nn.Conv2d(256, 1024, 1, 1),
           nn.BatchNorm2d(1024),
           nn.LeakyReLU(),
           #1024 x 1 x 1
           nn.Conv2d(1024, 4096, 1, 1),
           nn.BatchNorm2d(4096),
           nn.LeakyReLU(),
           nn.Conv2d(4096, 4096, 1, 1),
           nn.BatchNorm2d(4096),
           nn.LeakyReLU(),
           nn.Dropout2d(),
           #4096 x 1 x 1
           nn.Conv2d(4096, 16384, 1, 1),
           nn.BatchNorm2d(16384),
           nn.Tanh()
           #16384 x 1 x 1
        )

    def forward(self, input):
        output = input.view(-1, 100, 1, 1)
        output = self.main(output)
        output = output.view(-1, 4, 64, 64)
        return output 

class RectGenerator(nn.Module):
    def __init__(self, ngpu=1):
        super(RectGenerator, self).__init__()
        nz = 100
        ngf = 64
        nc = 4
        lin_hid_size = 3 * 2 * 4608
        self.ngpu = ngpu
        self.linear = nn.Sequential(
            nn.Linear(nz, lin_hid_size),
            nn.BatchNorm1d(lin_hid_size),
            nn.LeakyReLU()
        )
        self.main = nn.Sequential(
           #4608 x 3 x 2
            nn.ConvTranspose2d(4608, 2304, (4, 3), 1),
            nn.BatchNorm2d(2304),
            nn.LeakyReLU(),
            #2304 x 6 x 4
            nn.ConvTranspose2d(2304, 1152, 2, 2),
            nn.BatchNorm2d(1152),
            nn.LeakyReLU(),
            #1152 x 12 x 8
            nn.Conv2d(1152, 576, 1,1),
            nn.BatchNorm2d(576),
            nn.LeakyReLU(),
            #576 x 12 x 8
            nn.Conv2d(576, 288, 1, 1),
            nn.BatchNorm2d(288),
            nn.Tanh()
            # #288 x 12 x 8
        )

    def forward(self, input):
        output = self.linear(input)
        output = output.view(-1, 4608, 3, 2)
        output = self.main(output)
        return output

class RectDiscriminator(nn.Module):
    def __init__(self, ngpu=1):
        super(RectDiscriminator, self).__init__()
        nc = 288
        ndf = 2 * nc
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is 288 x 12 x 8
            nn.Conv2d(nc, ndf, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # # state size. 576 x 6 x 4
            nn.Conv2d(ndf, ndf * 2, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # # state size. 1152 x 3 x 2
            nn.Conv2d(ndf * 2, ndf * 4, (1, 2), (1, 2), bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # # state size. 2304 x 3 x 1
            nn.Conv2d(ndf * 4, ndf * 8, (2, 1), (2, 1), bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.4),
            # state size. 4608 x 1 x 1
        )
        self.out = nn.Sequential(
           nn.Linear(4608, 1),
           nn.Sigmoid()
        )
    def forward(self, input):
        output = self.main(input)
        if len(output.shape) < 4:
           output = output.view(1, -1)
        else:
           output = output.squeeze()
        output = self.out(output)
        if len(output.shape) < 2:
           return output
        return output.squeeze(1)
        # return output.view(-1, 1).squeeze(1)

class Discriminator(nn.Module):
    def __init__(self, ngpu=1):
        ndf = 128
        nc = 4
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        output = self.main(input)
        #ndf = 128
        return output.view(-1, 1).squeeze(1)
class NewDiscriminator(nn.Module):
    def __init__(self, ngpu=1):
        super(NewDiscriminator, self).__init__()
        ndf = 128
        nc = 4
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 2048, 4, 1, 0, bias=False),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.linear = nn.Sequential(
           nn.Linear(2048, 1),
           nn.Sigmoid()
        )
    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 2048)
        output = self.linear(output)
        #ndf = 128
        return output

class ResDiscriminator(nn.Module):
    def __init__(self, ngpu=1):
        super(ResDiscriminator, self).__init__()
        self.adapter = nn.Sequential(
                        nn.Conv2d(4, 3, 3, padding='same'),
                        nn.BatchNorm2d(3),
                        nn.ReLU()
            )
        self.res = resnet18(pretrained=False)
        self.out = nn.Sequential(
                          nn.Linear(1000, 1),
                        nn.Sigmoid()
                           )
  
    def forward(self, x):
        output = self.adapter(x)
        output = self.res(output)
        output = self.out(output)
        return output

     
class GAN(pl.LightningModule):

  def __init__(self):
    super().__init__()
    self.generator = Generator()
    self.discriminator = Discriminator()
    # After each epoch, we generate 100 images using the noise
    # vector here (self.test_noises). We save the output images
    # in a list (self.test_progression) for plotting later.
    self.test_noises = torch.randn(100,1,100, device=self.device)
    self.test_progression = []

  def forward(self, z):
    """
    Generates an image using the generator
    given input noise z
    """
    return self.generator(z)

  def generator_step(self, x):
    """
    Training step for generator
    1. Sample random noise
    2. Pass noise to generator to
       generate images
    3. Classify generated images using
       the discriminator
    4. Backprop loss to the generator
    """
    
    # Sample noise
    batch_size = x.shape[0] if x.shape[0] > 1 else 2
    z = torch.randn(batch_size, 100, device=self.device)

    # Generate images
    generated_imgs = self(z)

    # Classify generated images
    # using the discriminator
    d_output = torch.squeeze(self.discriminator(generated_imgs))
    if x.shape[0] <= 1:
       d_output = d_output[0]
    # Backprop loss. We want to maximize the discriminator's
    # loss, which is equivalent to minimizing the loss with the true
    # labels flipped (i.e. y_true=1 for fake images). We do this
    # as PyTorch can only minimize a function instead of maximizing
    g_loss = nn.BCELoss()(d_output,
                           torch.ones(x.shape[0], device=self.device))

    return g_loss

  def discriminator_step(self, x):
    """
    Training step for discriminator
    1. Get actual images
    2. Predict probabilities of actual images and get BCE loss
    3. Get fake images from generator
    4. Predict probabilities of fake images and get BCE loss
    5. Combine loss from both and backprop loss to discriminator
    """
    
    # Real images
    d_output = torch.squeeze(self.discriminator(x))
    loss_real = nn.BCELoss()(d_output,
                             torch.ones(x.shape[0], device=self.device))

    # Fake images
    z = torch.randn(x.shape[0], 100, device=self.device)
    generated_imgs = self(z)
    d_output = torch.squeeze(self.discriminator(generated_imgs))
    loss_fake = nn.BCELoss()(d_output,
                             torch.zeros(x.shape[0], device=self.device))

    return loss_real + loss_fake

  def training_step(self, batch, optimizer_idx):
    X = batch

    # train generator
    if optimizer_idx == 0:
      loss = self.generator_step(X)
    
    # train discriminator
    if optimizer_idx == 1:
      loss = self.discriminator_step(X)

    return loss

  def configure_optimizers(self):
    g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002)
    d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002)
    return [g_optimizer, d_optimizer]

  def training_epoch_end(self, training_step_outputs):
    epoch_test_images = self(self.test_noises)
    self.test_progression.append(epoch_test_images)