from torch import nn, optim
import torch

class MseKlLoss(nn.Module):
    def __init__(self, beta=1e-4):
        super(MseKlLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.beta = beta

    def kl_divergence(self, mu, logvar):
        # KL divergence between the learned distribution and standard normal distribution
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl

    def forward(self, yhat, y, mu, logvar):
        mse_loss = self.mse(yhat, y)
        kl_loss = self.kl_divergence(mu, logvar)
        combined_loss = mse_loss + self.beta * kl_loss
        return combined_loss
