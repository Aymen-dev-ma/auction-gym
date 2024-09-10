import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
import torch.nn.functional as F  # Add this line

class Encoder(nn.Module):
    def __init__(self, context_dim, item_dim, z_dim):
        super(Encoder, self).__init__()
        self.context_dim = context_dim
        self.item_dim = item_dim
        self.z_dim = z_dim

        self.fc1 = nn.Linear(self.context_dim + self.item_dim, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc31 = nn.Linear(1000, z_dim)  # Mean for latent variables
        self.fc32 = nn.Linear(1000, z_dim)  # Log variance for latent variables
        self.softplus = nn.Softplus()

    def forward(self, context, items):
        inputs = torch.cat((context, items), dim=-1)
        hidden1 = self.softplus(self.fc1(inputs))
        hidden2 = self.softplus(self.fc2(hidden1))
        z_loc = self.fc31(hidden2)
        z_scale = torch.exp(self.fc32(hidden2))
        return z_loc, z_scale

class Decoder(nn.Module):
    def __init__(self, context_dim, item_dim, z_dim):
        super(Decoder, self).__init__()
        hidden_dim = 1000
        self.fc1 = nn.Linear(z_dim + item_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, context_dim)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z, items):
        inputs = torch.cat((z, items), dim=-1)
        hidden1 = self.softplus(self.fc1(inputs))
        hidden2 = self.softplus(self.fc2(hidden1))
        hidden3 = self.softplus(self.fc3(hidden2))
        loc_context = self.sigmoid(self.fc4(hidden3))
        return loc_context

class CVAE(nn.Module):
    def __init__(self, context_dim, item_dim, z_dim, use_cuda=False):
        super(CVAE, self).__init__()
        self.context_dim = context_dim
        self.item_dim = item_dim
        self.z_dim = z_dim
        self.use_cuda = use_cuda

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(context_dim + item_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, z_dim)
        self.fc_logvar = nn.Linear(64, z_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(context_dim + z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, item_dim)
        )

    def encode(self, context, item):
        inputs = torch.cat([context, item], dim=1)
        h = self.encoder(inputs)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, context, z):
        inputs = torch.cat([context, z], dim=1)
        return self.decoder(inputs)

    def forward(self, context, item):
        mu, logvar = self.encode(context, item)
        z = self.reparameterize(mu, logvar)
        return self.decode(context, z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def sample(self, context, num_samples=1):
        if self.use_cuda:
            context = context.cuda()
        
        z = torch.randn(num_samples, self.z_dim)
        if self.use_cuda:
            z = z.cuda()
        
        context_repeated = context.repeat(num_samples, 1)
        samples = self.decode(context_repeated, z)
        return samples

class SCM():
    def __init__(self, vae, mu, sigma):
        self.vae = vae
        self.image_dim = vae.item_dim
        self.z_dim = vae.z_dim
        self.label_dims = vae.context_dim

        def f_X(Y, Z, N):
            zs = Z.cuda()
            ys = Y.cuda()
            p = vae.decoder(zs, ys)
            return (N < p.cpu()).type(torch.float)

        def f_Y(N):
            beta = 12
            indices = torch.tensor(range(N.size(0))).to(torch.float32)
            smax = F.softmax(beta*N, dim=0)
            argmax_ind = torch.sum(smax*indices)
            return argmax_ind

        def f_Z(N):
            return N * sigma + mu

        def model(noise):
            N_X = pyro.sample('N_X', noise['N_X'].to_event(1))
            N_Y = pyro.sample('N_Y', noise['N_Y'].to_event(1))
            N_Z = pyro.sample('N_Z', noise['N_Z'].to_event(1))

            Z = pyro.sample('Z', dist.Normal(f_Z(N_Z), 1e-1).to_event(1))
            Y = pyro.sample('Y', dist.Normal(f_Y(N_Y), 1e-1).to_event(1))
            X = pyro.sample('X', dist.Normal(f_X(Y, Z, N_X), 1e-2).to_event(1))

            noise_samples = N_X, N_Y, N_Z
            variable_samples = X, Y, Z
            return variable_samples, noise_samples

        self.model = model
        self.init_noise = {
            'N_X': dist.Uniform(torch.zeros(vae.item_dim), torch.ones(vae.item_dim)),
            'N_Z': dist.Normal(torch.zeros(vae.z_dim), torch.ones(vae.z_dim)),
            'N_Y': dist.Uniform(torch.zeros(vae.context_dim), torch.ones(vae.context_dim))
        }

    def __call__(self):
        return self.model(self.init_noise)

# Function to load CVAE and create SCM
def load_cvae_and_create_scm(cvae_path, context_dim, item_dim, z_dim):
    vae = CVAE(context_dim, item_dim, z_dim)
    vae.load_state_dict(torch.load(cvae_path))
    vae.eval()

    # Generate some data to get mu and sigma
    context = torch.randn(1, vae.context_dim)
    item = torch.randn(1, vae.item_dim)
    mu, logvar = vae.encode(context, item)
    sigma = torch.exp(0.5 * logvar)

    scm = SCM(vae, mu, sigma)
    return vae, scm