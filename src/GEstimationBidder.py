# GEstimationBidder.py
import numpy as np
import torch
from scipy.optimize import minimize
from Models import PyTorchLogisticRegression

class GEstimationBidder:
    def __init__(self, rng):
        self.rng = rng

    def bid(self, value, context, estimated_CTR):
        bid = value * estimated_CTR
        return bid

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        utilities = np.zeros_like(values)
        utilities[won_mask] = (values[won_mask] * outcomes[won_mask]) - prices[won_mask]
        utilities = torch.Tensor(utilities)

        X = np.hstack((estimated_CTRs.reshape(-1,1), values.reshape(-1,1)))
        X = torch.Tensor(X)
        y = won_mask.astype(np.uint8).reshape(-1,1)
        y = torch.Tensor(y)
        
        self.model = PyTorchLogisticRegression(n_dim=2, n_items=1)
        self.model.train()
        epochs = 8192 * 2
        lr = 2e-3
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, min_lr=1e-7, factor=0.1, verbose=True)

        losses = []
        for epoch in range(int(epochs)):
            optimizer.zero_grad()
            loss = self.model.loss(self.model(X).squeeze(), y.squeeze())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            scheduler.step(loss)

            if epoch > 1024 and np.abs(losses[-100] - losses[-1]) < 1e-6:
                print(f'Stopping at Epoch {epoch}')
                break

        self.model.eval()

