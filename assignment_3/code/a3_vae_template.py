import argparse

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from collections import OrderedDict

from datasets.bmnist import bmnist

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        input_size = 28*28

        ## Hidden layer is shared between mu and sigma
        self.hidden = nn.Sequential(OrderedDict([
          ('linear_hidden', nn.Linear(input_size, hidden_dim)),
          ('tanh_hidden', nn.Tanh()) ## maybe try RelU
        ]))

        self.mu = nn.Sequential(OrderedDict([
          ('output_mu', nn.Linear(hidden_dim, z_dim))
        ]))

        self.sigma = nn.Sequential(OrderedDict([
          ('output_sigma', nn.Linear(hidden_dim, z_dim)),
          ('relu_sigma_output', nn.ReLU())
        ]))

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """

        #######################
        # ENFORCE CONSTRAINTS?
        #######################

        hidden = self.hidden(input)
        mean = self.mu(hidden)
        std = self.sigma(hidden)

        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        output_size = 28*28

        self.decoder = nn.Sequential(OrderedDict([
          ('hidden', nn.Linear(z_dim, hidden_dim)),
          ('relu_hidden', nn.ReLU()),
          ('output', nn.Linear(hidden_dim, output_size)),
          ('sigmoid_output', nn.Sigmoid())
        ]))

    def forward(self, input):
        """
        Perform forward pass of decoder.

        Returns mean with shape [batch_size, 784].
        """
        mean = self.decoder(input)

        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        mean, std = self.encoder(input)
        epsilon = torch.randn_like(std)
        z = mean + std * epsilon
        reconstruction = self.decoder(z)
        
        # Log stability
        stability = 1e-8

        L_recon = -(input * torch.log(reconstruction) + (1 - input) * torch.log(1 - reconstruction)).sum(dim=1)
        L_reg = 0.5 * (std**2 + mean**2 - 1 - torch.log(std**2 + stability)).sum(dim=1)

        # print("L_recon", L_recon.mean(dim=0))
        # print("L_reg", L_reg.mean(dim=0))

        average_negative_elbo = (L_recon+L_reg).mean(dim=0)

        return average_negative_elbo, (L_recon.mean(dim=0), L_reg.mean(dim=0))

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        # noise = torch.randn(n_samples, self.z_dim)).cuda()
        # probabilities = self.decoder(noise)
        # bernoullis = [torch.distributions.Bernoulli(p) for p in probabilities]
        # sampled_ims = [bernoulli.sample() for bernoulli in bernoullis]

        noise = torch.randn(1, self.z_dim).to(DEVICE)
        probabilities = self.decoder(noise)
        bernoulli = torch.distributions.Bernoulli(probabilities)
        sampled_ims = bernoulli.sample(n_samples)
        im_means = sampled_ims.mean(dim=0)

        return sampled_ims, im_means


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    elbo_f = train if model.training else test
    elbos = elbo_f(model, data, optimizer)
    average_epoch_elbo = torch.stack(elbos).mean(dim=0)

    return average_epoch_elbo

def train(model, data, optimizer):
    train_elbos = []
    for batch in iter(data):
        optimizer.zero_grad()
        train_elbo, losses = model(batch.view(-1, 28*28).to(DEVICE))
        train_elbos.append(train_elbo)
        train_elbo.backward()
        optimizer.step()

    print("Lrecon, Lreg", losses)
    print("loss", train_elbo.item())

    return train_elbos

def test(model, data, optimizer):
    test_elbos = []
    with torch.no_grad(): 
        for batch in iter(data):
            test_elbo, losses = model(batch.view(-1, 28*28).to(DEVICE))
            test_elbos.append(test_elbo)
    
    return test_elbos


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def main():
    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())

    train_curve, val_curve = [], []
    samples = []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------
        img = torchvision.utils.make_grid(batch, nrow=5)

    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------

    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()

    main()
