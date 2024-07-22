import numpy as np
from dataclasses import dataclass

@dataclass
class ImpressionOpportunity:
    context: np.array
    item: np.uint32
    value: np.float32
    bid: np.float32
    best_expected_value: np.float32
    true_CTR: np.float32
    estimated_CTR: np.float32
    price: np.float32
    second_price: np.float32
    outcome: bool
    won: bool

    def set_true_CTR(self, best_expected_value, true_CTR):
        self.best_expected_value = best_expected_value
        self.true_CTR = true_CTR

    def set_price_outcome(self, price, second_price, outcome, won=True):
        self.price = price
        self.second_price = second_price
        self.outcome = outcome
        self.won = won

    def set_price(self, price):
        self.price = price
