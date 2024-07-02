import torch
import torch.nn as nn
import torch.optim as optim
from Bidder import PolicyLearningBidder

class PolicyModelWithCausalInference(nn.Module):
    def __init__(self, input_dim):
        super(PolicyModelWithCausalInference, self).__init__()
        self.shared_linear = nn.Linear(input_dim, 64)
        self.mu_linear_out = nn.Linear(64, 1)
        self.sigma_linear_out = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.nn.functional.softplus(self.shared_linear(x))
        mu = torch.nn.functional.softplus(self.mu_linear_out(x))
        sigma = torch.nn.functional.softplus(self.sigma_linear_out(x))
        return mu, sigma

    def initialise_policy(self, observed_contexts, gammas):
        observed_contexts = torch.Tensor(observed_contexts)  # Ensure inputs are tensors
        predicted_mu_gammas = torch.nn.functional.softplus(self.mu_linear_out(torch.nn.functional.softplus(self.shared_linear(observed_contexts))))
        predicted_sigma_gammas = torch.nn.functional.softplus(self.sigma_linear_out(torch.nn.functional.softplus(self.shared_linear(observed_contexts))))
        return predicted_mu_gammas, predicted_sigma_gammas

class PolicyLearningBidderWithCausalInference(PolicyLearningBidder):
    def __init__(self, rng, gamma_sigma, init_gamma, loss):
        super().__init__(rng, gamma_sigma, init_gamma, loss)
        self.model = PolicyModelWithCausalInference(input_dim=5)  # Change input_dim to match your context dimension

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, agent_name):
        # Convert numpy arrays to tensors
        contexts = torch.Tensor(contexts)
        values = torch.Tensor(values)
        bids = torch.Tensor(bids)
        prices = torch.Tensor(prices)
        outcomes = torch.Tensor(outcomes)
        estimated_CTRs = torch.Tensor(estimated_CTRs)
        won_mask = torch.Tensor(won_mask)

        # Compute importance weights for IPTW
        importance_weights = torch.div(estimated_CTRs, self.logging_propensity(contexts))

        # Compute doubly robust estimates
        dr_estimates = (outcomes * importance_weights) - ((1 - outcomes) * (1 - importance_weights))

        # Update the policy using the doubly robust estimates
        self.model.initialise_policy(contexts, torch.Tensor(self.gammas))
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        for epoch in range(100):
            optimizer.zero_grad()
            predictions, _ = self.model(contexts)  # Obtain predictions for mu
            loss = loss_fn(predictions, dr_estimates.unsqueeze(1))  # Ensure dimensions match
            loss.backward()
            optimizer.step()

            if epoch % 50 == 0:
                print(f'Epoch {epoch}: Loss = {loss.item()}')

        if plot:
            self.plot_gammas(figsize, fontsize, agent_name, iteration)

    def logging_propensity(self, contexts):
        # Placeholder for logging propensity function
        return torch.ones(contexts.size(0)) * 0.5

    def plot_gammas(self, figsize, fontsize, agent_name, iteration):
        import matplotlib.pyplot as plt
        plt.figure(figsize=figsize)
        plt.plot(self.gammas)
        plt.title(f'Gamma Values for {agent_name} at Iteration {iteration}', fontsize=fontsize)
        plt.xlabel('Index', fontsize=fontsize)
        plt.ylabel('Gamma Value', fontsize=fontsize)
        plt.show()
