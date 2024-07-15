import torch
import torch.nn as nn
import torch.nn.functional as F

class VAELSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAELSTM, self).__init__()
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder_lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h, _ = self.encoder_lstm(x)
        h = h[:, -1, :]  # Take the last hidden state
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = z.unsqueeze(1).repeat(1, 10, 1)  # Assuming a sequence length of 10 for the decoder
        h, _ = self.decoder_lstm(z)
        return self.fc_out(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

# For the LSTM part used in bidding
class LSTMBidder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(LSTMBidder, self).__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h, _ = self.lstm(x)
        h = h[:, -1, :]  # Take the last hidden state
        return F.softmax(self.fc(h), dim=-1)
