import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity

        self.linear_layer1 = nn.Sequential(OrderedDict([
          ('linear1', nn.Linear(args.latent_dim, 128)),
          ('lrelu1', nn.LeakyReLU(0.2))
        ]))
        self.linear_layer2 = nn.Sequential(OrderedDict([
          ('linear2', nn.Linear(128, 256)),
          ('batch2', nn.BatchNorm1d(256)),
          ('lrelu2', nn.LeakyReLU(0.2))
        ]))
        self.linear_layer3 = nn.Sequential(OrderedDict([
          ('linear3', nn.Linear(256, 512)),
          ('batch3', nn.BatchNorm1d(512)),
          ('lrelu3', nn.LeakyReLU(0.2))
        ]))
        self.linear_layer4 = nn.Sequential(OrderedDict([
          ('linear4', nn.Linear(512, 1024)),
          ('batch4', nn.BatchNorm1d(1024)),
          ('lrelu4', nn.LeakyReLU(0.2))
        ]))
        self.linear_layer5 = nn.Sequential(OrderedDict([
          ('linear5', nn.Linear(1024, 28*28)),
          ('batch5', nn.BatchNorm1d(28*28)),
          ('tanh5', nn.Tanh())
        ]))

    def forward(self, z):
       x = self.linear_layer1(z)
       x = self.linear_layer2(x)
       x = self.linear_layer3(x)
       x = self.linear_layer4(x)
       image = self.linear_layer5(x)

       return image

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity

        self.linear_layer1 = nn.Sequential(OrderedDict([
          ('linear1', nn.Linear(28*28, 512)),
          ('lrelu1', nn.LeakyReLU(0.2))
        ]))
        self.linear_layer2 = nn.Sequential(OrderedDict([
          ('linear2', nn.Linear(512, 256)),
          ('lrelu2', nn.LeakyReLU(0.2))
        ]))
        self.linear_layer3 = nn.Sequential(OrderedDict([
          ('linear3', nn.Linear(256, 1)),
          ('sigmoid3', nn.Sigmoid())
        ]))

    def forward(self, img):
       x = self.linear_layer1(img)
       x = self.linear_layer2(x)
       decision = self.linear_layer3(x)
       
       return decision

def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):

    D = discriminator.to(DEVICE)
    G = generator.to(DEVICE)

    # Log stability
    # stability = 1e-8
    # max_loss = 1000
    G_losses_epoch = []
    D_losses_epoch = []    

    for epoch in range(args.n_epochs):

        G_losses_batch = []
        D_losses_batch = []

        for i, (imgs, _) in enumerate(dataloader):

            batch = imgs.view(-1, 28*28).to(DEVICE)
            z = torch.randn((batch.shape[0], args.latent_dim)).to(DEVICE)
            samples = G(z)
            criterion = nn.BCELoss()

            # Train Generator
            optimizer_G.zero_grad()
            G_targets = torch.ones(batch.shape[0]).to(DEVICE)
            G_loss = criterion(D(samples), G_targets)
            # G_loss.clamp(min=stability, max=max_loss)
            G_losses_batch.append(G_loss.item())
            G_loss.backward(retain_graph=True)
            optimizer_G.step()
            
            # Train Discriminator
            optimizer_D.zero_grad()
            D_targets_real = torch.uniform_(torch.ones(batch.shape[0]), low=0.7, high=1.3).to(DEVICE)
            D_targets_fake = torch.uniform_(torch.zeros(batch.shape[0]), low=-0.3, high=0.3).to(DEVICE)
            D_loss = criterion(D(batch), D_targets_real) + criterion(D(samples), D_targets_fake)
            # D_loss.clamp(min=stability, max=max_loss)
            D_losses_batch.append(D_loss.item())
            D_loss.backward()
            optimizer_D.step()

            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                save_image(samples.view(-1, 1, 28, 28)[:25],
                           'images/{}.png'.format(batches_done),
                           nrow=5, normalize=True)

        G_loss_epoch = np.mean(G_losses_batch)
        D_loss_epoch = np.mean(D_losses_batch)
        G_losses_epoch.append(G_loss_epoch)
        D_losses_epoch.append(D_loss_epoch)
        print("epoch {}...".format(epoch))
        print("G_loss_epoch", G_loss_epoch)
        print("D_loss_epoch", D_loss_epoch)
    
    save_training_plot(G_losses_epoch, D_losses_epoch, "gan_training.pdf")


def save_training_plot(G_losses, D_losses, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(G_losses, label='generator loss')
    plt.plot(D_losses, label='discriminator loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.tight_layout()
    plt.savefig(filename)

def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,),
                                                (0.5,))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator()
    discriminator = Discriminator()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    args = parser.parse_args()

    main()
