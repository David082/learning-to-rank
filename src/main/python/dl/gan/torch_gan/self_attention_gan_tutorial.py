# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/12/25
version :
refer :
https://github.com/torchgan/torchgan/tree/master/tutorials
"""
# General Imports
import os
import random
import matplotlib.pyplot as plt
import numpy as np
# Pytorch and Torchvision Imports
import torch
import torchvision
from torch.optim import Adam
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as vutils
# Torchgan Imports
import torchgan
from torchgan.layers import SpectralNorm2d, SelfAttention2d
from torchgan.models import Generator, Discriminator
from torchgan.losses import WassersteinGeneratorLoss, WassersteinDiscriminatorLoss, WassersteinGradientPenalty
from torchgan.trainer import Trainer


class SAGANGenerator(Generator):
    def __init__(self, encoding_dims=100, step_channels=64):
        super(SAGANGenerator, self).__init__(encoding_dims, 'none')
        d = int(step_channels * 8)
        self.model = nn.Sequential(
            SpectralNorm2d(nn.ConvTranspose2d(self.encoding_dims, d, 4, 1, 0, bias=True)),
            nn.BatchNorm2d(d), nn.LeakyReLU(0.2),
            SpectralNorm2d(nn.ConvTranspose2d(d, d // 2, 4, 2, 1, bias=True)),
            nn.BatchNorm2d(d // 2), nn.LeakyReLU(0.2),
            SpectralNorm2d(nn.ConvTranspose2d(d // 2, d // 4, 4, 2, 1, bias=True)),
            nn.BatchNorm2d(d // 4), nn.LeakyReLU(0.2),
            SelfAttention2d(d // 4),
            SpectralNorm2d(nn.ConvTranspose2d(d // 4, d // 8, 4, 2, 1, bias=True)),
            nn.BatchNorm2d(d // 8),
            SelfAttention2d(d // 8),
            SpectralNorm2d(nn.ConvTranspose2d(d // 8, 3, 4, 2, 1, bias=True)), nn.Tanh())

    def forward(self, x):
        x = x.view(-1, x.size(1), 1, 1)
        return self.model(x)


class SAGANDiscriminator(Discriminator):
    def __init__(self, step_channels=64):
        super(SAGANDiscriminator, self).__init__(3, 'none')
        d = step_channels
        self.model = nn.Sequential(
            SpectralNorm2d(nn.Conv2d(self.input_dims, d, 4, 2, 1, bias=True)),
            nn.BatchNorm2d(d), nn.LeakyReLU(0.2),
            SpectralNorm2d(nn.Conv2d(d, d * 2, 4, 2, 1, bias=True)),
            nn.BatchNorm2d(d * 2), nn.LeakyReLU(0.2),
            SpectralNorm2d(nn.Conv2d(d * 2, d * 4, 4, 2, 1, bias=True)),
            nn.BatchNorm2d(d * 4), nn.LeakyReLU(0.2),
            SelfAttention2d(d * 4),
            SpectralNorm2d(nn.Conv2d(d * 4, d * 8, 4, 2, 1, bias=True)),
            nn.BatchNorm2d(d * 8),
            SelfAttention2d(d * 8),
            SpectralNorm2d(nn.Conv2d(d * 8, 1, 4, 1, 0, bias=True)), nn.Tanh())

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    # Set random seed for reproducibility
    manualSeed = 144
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    print("Random Seed: ", manualSeed)

    # LOAD THE DATASET
    dataset = dsets.ImageFolder("./CelebA",
                                transform=transforms.Compose([transforms.CenterCrop(160),
                                                              transforms.Resize((64, 64)),
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    dataloader = data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)

    # MODEL PARAMETERS & HYPERPARAMETERS
    network_params = {
        "generator": {"name": SAGANGenerator, "args": {"step_channels": 32},
                      "optimizer": {"name": Adam, "args": {"lr": 0.0001, "betas": (0.0, 0.999)}}},
        "discriminator": {"name": SAGANDiscriminator, "args": {"step_channels": 32},
                          "optimizer": {"name": Adam, "args": {"lr": 0.0004, "betas": (0.0, 0.999)}}}
    }
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        # Use deterministic cudnn algorithms
        torch.backends.cudnn.deterministic = True
        epochs = 20
    else:
        device = torch.device("cpu")
        epochs = 10
    print("Device: {}".format(device))
    print("Epochs: {}".format(epochs))

    losses_list = [WassersteinGeneratorLoss(), WassersteinDiscriminatorLoss(clip=(-0.01, 0.01))]

    # VISUALIZE THE TRAINING DATA
    # Plot some of the training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()

    # TRAINING THE GENERATOR & DISCRIMINATOR
    trainer = Trainer(network_params, losses_list, sample_size=64, epochs=epochs, device=device)
    trainer(dataloader)
    trainer.complete()

    # VISUALIZE THE GENERATED DATA
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(plt.imread("{}/epoch{}_generator.png".format(trainer.recon, trainer.epochs)))
    plt.show()
