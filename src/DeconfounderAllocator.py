# DeconfounderAllocator.py
import numpy as np
import torch
from Models import PyTorchLogisticRegression

class DeconfounderAllocator:
    def __init__(self, rng, embedding_size, num_items, thompson_sampling=True):
        self.response_model = PyTorchLogisticRegression(n_dim=embedding_size, n_items=num_items)
        self.thompson_sampling = thompson_sampling
        self.rng = rng

    def update(self, contexts, items, outcomes, iteration, plot, figsize, fontsize, name):
        if len(outcomes) < 2:
            return

        self.response_model.train()
        epochs = 8192 * 2
        lr = 2e-3
        optimizer = torch.optim.Adam(self.response_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)

        contexts, items, outcomes = torch.Tensor(contexts), torch.LongTensor(items), torch.Tensor(outcomes)
        losses = []
        for epoch in range(int(epochs)):
            optimizer.zero_grad()
            loss = self.response_model.loss(torch.squeeze(self.response_model.predict_item(contexts, items)), outcomes)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            scheduler.step(loss)
            if epoch > 1024 and np.abs(losses[-100] - losses[-1]) < 1e-6:
                print(f'Stopping at Epoch {epoch}')
                break

        with torch.no_grad():
            for item in range(self.response_model.m.shape[0]):
                item_mask = items == item
                contexts_item = torch.Tensor(contexts[item_mask])
                self.response_model.laplace_approx(contexts_item, item)
            self.response_model.update_prior()

        self.response_model.eval()

    def estimate_CTR(self, context, sample=True):
        return self.response_model(torch.from_numpy(context.astype(np.float32)), sample=(self.thompson_sampling and sample)).detach().numpy()
