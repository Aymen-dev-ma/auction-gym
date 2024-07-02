import torch
import numpy as np
from tqdm import tqdm
from Models import BidShadingContextualBandit, PyTorchWinRateEstimator
from Bidder import Bidder
from causal_inference import estimate_propensity_scores, doubly_robust_estimation

class PolicyLearningBidderWithCausalInference(Bidder):
    def __init__(self, rng, gamma_sigma, loss, init_gamma=1.0):
        super(PolicyLearningBidderWithCausalInference, self).__init__(rng)
        self.gamma_sigma = gamma_sigma
        self.prev_gamma = init_gamma
        self.gammas = []
        self.propensities = []
        self.model = BidShadingContextualBandit(loss)
        self.model_initialised = False
        self.truthful = False  # Default
        self.winrate_model = PyTorchWinRateEstimator()

    def bid(self, value, context, estimated_CTR):
        bid = value * estimated_CTR
        if not self.model_initialised:
            gamma = self.rng.normal(self.prev_gamma, self.gamma_sigma)
            propensity = self._normal_pdf(gamma)
        else:
            x = torch.Tensor([estimated_CTR, value])
            gamma, propensity = self.model(x)
            gamma = torch.clip(gamma, 0.0, 1.0)
        bid *= gamma.detach().item() if self.model_initialised else gamma
        self.gammas.append(gamma)
        self.propensities.append(propensity)
        return bid

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        utilities = self._compute_utilities(values, prices, outcomes, won_mask)
        X = np.hstack((estimated_CTRs.reshape(-1, 1), values.reshape(-1, 1)))

        if not self.model_initialised:
            self.model.initialise_policy(X, torch.Tensor(self.gammas))

        propensities = estimate_propensity_scores(X, bids)
        estimated_utilities = self._estimate_utilities(contexts, values, estimated_CTRs, won_mask)

        self._train_doubly_robust_policy(X, utilities, estimated_utilities, propensities, won_mask, iteration, plot, figsize, fontsize, name)

        self.model_initialised = True
        self.model.model_initialised = True

    def _compute_utilities(self, values, prices, outcomes, won_mask):
        utilities = np.zeros_like(values)
        utilities[won_mask] = (values[won_mask] * outcomes[won_mask]) - prices[won_mask]
        return torch.Tensor(utilities)

    def _estimate_utilities(self, contexts, values, estimated_CTRs, won_mask):
        gammas_numpy = np.array([g.detach().item() if self.model_initialised else g for g in self.gammas])
        orig_features = torch.Tensor(np.hstack((estimated_CTRs.reshape(-1, 1), values.reshape(-1, 1), gammas_numpy.reshape(-1, 1))))
        W = self.winrate_model(orig_features).squeeze().detach().numpy()

        V = estimated_CTRs * values
        P = estimated_CTRs * values * gammas_numpy
        estimated_utilities = W * (V - P)
        return torch.Tensor(estimated_utilities)

    def _train_doubly_robust_policy(self, X, utilities, estimated_utilities, propensities, won_mask, iteration, plot, figsize, fontsize, name):
        self.model.train()
        epochs = 8192 * 2
        lr = 2e-3
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, min_lr=1e-8, factor=0.2, verbose=True)

        losses = []
        best_epoch, best_loss = -1, np.inf
        for epoch in tqdm(range(int(epochs)), desc=f'{name}'):
            optimizer.zero_grad()
            loss = self.model.loss(X, torch.Tensor(self.gammas), propensities, utilities, utility_estimates=estimated_utilities, winrate_model=self.winrate_model, importance_weight_clipping_eps=50.0)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            scheduler.step(loss)

            if (best_loss - losses[-1]) > 1e-6:
                best_epoch = epoch
                best_loss = losses[-1]
            elif epoch - best_epoch > 512:
                print(f'Stopping at Epoch {epoch}')
                break

        self.model.eval()

    def _normal_pdf(self, g):
        return np.exp(-((self.prev_gamma - g) / self.gamma_sigma) ** 2 / 2) / (self.gamma_sigma * np.sqrt(2 * np.pi))
