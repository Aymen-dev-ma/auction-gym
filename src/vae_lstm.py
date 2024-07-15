import torch
import torch.nn as nn
import torch.nn.functional as F

class VAELSTM(nn.Module):
    def __init__(self, vae_input_dim, vae_hidden_dim, vae_latent_dim):
        super(VAELSTM, self).__init__()
        self.encoder = nn.LSTM(vae_input_dim, vae_hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(vae_hidden_dim, vae_latent_dim)
        self.fc_logvar = nn.Linear(vae_hidden_dim, vae_latent_dim)
        self.decoder = nn.LSTM(vae_latent_dim, vae_hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(vae_hidden_dim, vae_input_dim)

    def encode(self, x):
        _, (h_n, _) = self.encoder(x)
        h_n = h_n.squeeze(0)
        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = z.unsqueeze(1).repeat(1, self.decoder.input_size, 1)
        decoded, _ = self.decoder(z)
        decoded = self.fc_out(decoded)
        return decoded

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
