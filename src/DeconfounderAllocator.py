import torch
from torch import nn
import numpy as np
from Models import PyTorchLogisticRegression

class DeconfounderAllocator:
    def __init__(self, rng, embedding_size, num_items):
        self.model = PyTorchLogisticRegression(n_dim=embedding_size, n_items=num_items)
        self.rng = rng

    def estimate_CTR(self, context, sample=True):
        return self.model(torch.from_numpy(context.astype(np.float32)), sample=sample).detach().numpy()

    def update(self, contexts, items, outcomes, iteration, plot, figsize, fontsize, name):
        if len(outcomes) < 2:
            return
        self.model.train()
        epochs = 8192 * 2
        lr = 2e-3
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)
        contexts, items, outcomes = torch.Tensor(contexts), torch.LongTensor(items), torch.Tensor(outcomes)
        losses = []
        for epoch in range(int(epochs)):
            optimizer.zero_grad()
            loss = self.model.loss(torch.squeeze(self.model.predict_item(contexts, items)), outcomes)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            scheduler.step(loss)
            if epoch > 1024 and np.abs(losses[-100] - losses[-1]) < 1e-6:
                break
        with torch.no_grad():
            for item in range(self.model.m.shape[0]):
                item_mask = items == item
                context_item = torch.Tensor(contexts[item_mask])
                self.model.laplace_approx(context_item, item)
            self.model.update_prior()
        self.model.eval()
