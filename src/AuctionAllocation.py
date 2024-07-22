# AuctionAllocation.py
import numpy as np

class AllocationMechanism:
    def allocate(self, bids, num_slots):
        pass

class SecondPrice(AllocationMechanism):
    def allocate(self, bids, num_slots):
        winners = np.argsort(-bids)[:num_slots]
        prices = -np.sort(-bids)[1:num_slots+1]
        return winners, prices, prices
