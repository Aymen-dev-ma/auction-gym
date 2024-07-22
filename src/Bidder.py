# Bidder.py
import numpy as np
import torch
from Models import PyTorchLogisticRegression

class Bidder:
    def __init__(self, rng):
        self.rng = rng
        self.truthful = False

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        pass

    def clear_logs(self, memory):
        pass

class TruthfulBidder(Bidder):
    def __init__(self, rng):
        super(TruthfulBidder, self).__init__(rng)
        self.truthful = True

    def bid(self, value, context, estimated_CTR):
        return value * estimated_CTR

class EmpiricalShadedBidder(Bidder):
    def __init__(self, rng, gamma_sigma, init_gamma=1.0):
        self.gamma_sigma = gamma_sigma
        self.prev_gamma = init_gamma
        self.gammas = []
        super(EmpiricalShadedBidder, self).__init__(rng)

    def bid(self, value, context, estimated_CTR):
        bid = value * estimated_CTR
        gamma = self.rng.normal(self.prev_gamma, self.gamma_sigma)
        gamma = np.clip(gamma, 0.0, 1.0)
        bid *= gamma
        self.gammas.append(gamma)
        return bid

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        utilities = np.zeros_like(values)
        utilities[won_mask] = (values[won_mask] * outcomes[won_mask]) - prices[won_mask]

        gammas = np.array(self.gammas)

        if plot:
            _, _ = plt.subplots(figsize=figsize)
            plt.title('Raw observations', fontsize=fontsize + 2)
            plt.scatter(gammas, utilities, alpha=.25)
            plt.xlabel(r'Shading factor ($\gamma$)', fontsize=fontsize)
            plt.ylabel('Net Utility', fontsize=fontsize)
            plt.xticks(fontsize=fontsize - 2)
            plt.yticks(fontsize=fontsize - 2)

        min_gamma, max_gamma = np.min(gammas), np.max(gammas)
        grid_delta = .005
        num_buckets = int((max_gamma - min_gamma) // grid_delta) + 1
        buckets = np.linspace(min_gamma, max_gamma, num_buckets)
        x = []
        estimated_y_mean = []
        estimated_y_stderr = []
        bucket_lo = buckets[0]
        for idx, bucket_hi in enumerate(buckets[1:]):
            x.append((bucket_hi - bucket_lo) / 2.0 + bucket_lo)
            mask = np.logical_and(gammas < bucket_hi, bucket_lo <= gammas)
            num_samples = len(utilities[mask])
            if num_samples > 1:
                bucket_utility = utilities[mask].mean()
                estimated_y_mean.append(bucket_utility)
                estimated_y_stderr.append(np.std(utilities[mask]) / np.sqrt(num_samples))
            else:
                estimated_y_mean.append(np.nan)
                estimated_y_stderr.append(np.nan)
            bucket_lo = bucket_hi
        x = np.asarray(x)
        estimated_y_mean = np.asarray(estimated_y_mean)
        estimated_y_stderr = np.asarray(estimated_y_stderr)
        critical_value = 1.96
        U_lower_bound = estimated_y_mean - critical_value * estimated_y_stderr

        best_idx = len(x) - np.nanargmax(U_lower_bound[::-1]) - 1
        best_gamma = x[best_idx]
        best_gamma = np.clip(best_gamma, 0.0, 1.0)
        self.prev_gamma = best_gamma

        if plot:
            fig, axes = plt.subplots(figsize=figsize)
            plt.suptitle(name, fontsize=fontsize + 2)
            plt.title(f'Iteration: {iteration}', fontsize=fontsize)
            plt.plot(x, estimated_y_mean, label='Estimate', ls='--', color='red')
            plt.fill_between(x, estimated_y_mean - critical_value * estimated_y_stderr, estimated_y_mean + critical_value * estimated_y_stderr, alpha=.25, color='red', label='C.I.')
            plt.axvline(best_gamma, ls='--', color='gray', label='Best')
            plt.axhline(0, ls='-.', color='gray')
            plt.xlabel(r'Multiplicative Bid Shading Factor ($\gamma$)', fontsize=fontsize)
            plt.ylabel('Estimated Net Utility', fontsize=fontsize)
            plt.ylim(-1.0, 2.0)
            plt.xticks(fontsize=fontsize - 2)
            plt.yticks(fontsize=fontsize - 2)
            plt.legend(fontsize=fontsize)
            plt.tight_layout()

    def clear_logs(self, memory):
        if not memory:
            self.gammas = []
        else:
            self.gammas = self.gammas[-memory:]
