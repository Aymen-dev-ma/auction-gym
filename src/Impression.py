from dataclasses import dataclass
import numpy as np

@dataclass
class ImpressionOpportunity:
    context: np.ndarray
    item: int
    value: float
    bid: float
    best_expected_value: float
    true_CTR: float
    estimated_CTR: float
    price: float
    second_price: float
    outcome: bool
    won: bool

    def set_true_CTR(self, best_expected_value, true_CTR):
        self.best_expected_value = best_expected_value  # Best possible CTR (to compute regret from ad allocation)
        self.true_CTR = true_CTR  # True CTR for the chosen ad

    def set_price_outcome(self, price, second_price, outcome, won=True):
        self.price = price
        self.second_price = second_price
        self.outcome = outcome
        self.won = won

    def set_price(self, price):
        self.price = price
