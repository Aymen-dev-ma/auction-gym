import numpy as np
from DeconfounderAllocator import DeconfounderAllocator
from Impression import ImpressionOpportunity
from Models import sigmoid

class RLAgent:
    def __init__(self, rng, name, num_items, item_values, allocator, bidder, memory=0):
        self.rng = rng
        self.name = name
        self.num_items = num_items
        self.item_values = item_values
        self.net_utility = .0
        self.gross_utility = .0
        self.logs = []
        self.allocator = allocator
        self.bidder = bidder
        self.memory = memory

    def select_item(self, context):
        estim_CTRs = self.allocator.estimate_CTR(context)
        estim_values = estim_CTRs * self.item_values
        best_item = np.argmax(estim_values)
        if isinstance(self.allocator, DeconfounderAllocator) and self.allocator.thompson_sampling:
            estim_CTRs_MAP = self.allocator.estimate_CTR(context, sample=False)
            return best_item, estim_CTRs_MAP[best_item]
        return best_item, estim_CTRs[best_item]

    def bid(self, context):
        best_item, estimated_CTR = self.select_item(context)
        value = self.item_values[best_item]
        bid = self.bidder.bid(value, context, estimated_CTR)
        self.logs.append(ImpressionOpportunity(context=context,
                                               item=best_item,
                                               estimated_CTR=estimated_CTR,
                                               value=value,
                                               bid=bid,
                                               best_expected_value=0.0,
                                               true_CTR=0.0,
                                               price=0.0,
                                               second_price=0.0,
                                               outcome=0,
                                               won=False))
        return bid, best_item

    def charge(self, price, second_price, outcome):
        self.logs[-1].set_price_outcome(price, second_price, outcome, won=True)
        last_value = self.logs[-1].value * outcome
        self.net_utility += (last_value - price)
        self.gross_utility += last_value

    def set_price(self, price):
        self.logs[-1].set_price(price)

    def update(self, iteration, plot=False, figsize=(8,5), fontsize=14):
        contexts = np.array([opp.context for opp in self.logs])
        items = np.array([opp.item for opp in self.logs])
        values = np.array([opp.value for opp in self.logs])
        bids = np.array([opp.bid for opp in self.logs])
        prices = np.array([opp.price for opp in self.logs])
        outcomes = np.array([opp.outcome for opp in self.logs])
        estimated_CTRs = np.array([opp.estimated_CTR for opp in self.logs])
        won_mask = np.array([opp.won for opp in self.logs])
        self.allocator.update(contexts[won_mask], items[won_mask], outcomes[won_mask], iteration, plot, figsize, fontsize, self.name)
        self.bidder.update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, self.name)

    def get_allocation_regret(self):
        return np.sum([opp.best_expected_value - opp.true_CTR * opp.value for opp in self.logs])

    def get_estimation_regret(self):
        return np.sum([opp.estimated_CTR * opp.value - opp.true_CTR * opp.value for opp in self.logs])

    def get_overbid_regret(self):
        return np.sum([(opp.price - opp.second_price) * opp.won for opp in self.logs])

    def get_underbid_regret(self):
        return np.sum([(opp.price - opp.bid) * (not opp.won) * (opp.price < (opp.true_CTR * opp.value)) for opp in self.logs])

    def get_CTR_RMSE(self):
        return np.sqrt(np.mean([(opp.true_CTR - opp.estimated_CTR)**2 for opp in self.logs]))

    def get_CTR_bias(self):
        return np.mean([(opp.estimated_CTR / opp.true_CTR) for opp in filter(lambda opp: opp.won, self.logs)])

    def clear_utility(self):
        self.net_utility = .0
        self.gross_utility = .0

    def clear_logs(self):
        if not self.memory:
            self.logs = []
        else:
            self.logs = self.logs[-self.memory:]
        self.bidder.clear_logs(memory=self.memory)
