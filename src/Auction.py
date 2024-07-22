import numpy as np
from AuctionAllocation import AllocationMechanism, FirstPrice, SecondPrice
from Impression import ImpressionOpportunity
from Bidder import Bidder
from BidderAllocation import OracleAllocator
from Models import sigmoid

class Auction:
    def __init__(self, rng, allocation, agents, agent2items, agents2item_values, max_slots, embedding_size, embedding_var, obs_embedding_size, num_participants_per_round):
        self.rng = rng
        self.allocation = allocation
        self.agents = agents
        self.max_slots = max_slots
        self.revenue = .0
        self.agent2items = agent2items
        self.agents2item_values = agents2item_values
        self.embedding_size = embedding_size
        self.embedding_var = embedding_var
        self.obs_embedding_size = obs_embedding_size
        self.num_participants_per_round = num_participants_per_round

    def simulate_opportunity(self):
        num_slots = self.rng.integers(1, self.max_slots + 1)
        true_context = np.concatenate((self.rng.normal(0, self.embedding_var, size=self.embedding_size), [1.0]))
        obs_context = np.concatenate((true_context[:self.obs_embedding_size], [1.0]))
        bids = []
        CTRs = []
        participating_agents_idx = self.rng.choice(len(self.agents), self.num_participants_per_round, replace=False)
        participating_agents = [self.agents[idx] for idx in participating_agents_idx]
        for agent in participating_agents:
            if isinstance(agent.allocator, OracleAllocator):
                bid, item = agent.bid(true_context)
            else:
                bid, item = agent.bid(obs_context)
            bids.append(bid)
            true_CTR = sigmoid(true_context @ self.agent2items[agent.name].T)
            agent.logs[-1].set_true_CTR(np.max(true_CTR * self.agents2item_values[agent.name]), true_CTR[item])
            CTRs.append(true_CTR[item])
        bids = np.array(bids)
        CTRs = np.array(CTRs)
        winners, prices, second_prices = self.allocation.allocate(bids, num_slots)
        outcomes = self.rng.binomial(1, CTRs[winners])
        for slot_id, (winner, price, second_price, outcome) in enumerate(zip(winners, prices, second_prices, outcomes)):
            for agent_id, agent in enumerate(participating_agents):
                if agent_id == winner:
                    agent.charge(price, second_price, bool(outcome))
                else:
                    agent.set_price(price)
            self.revenue += price

    def clear_revenue(self):
        self.revenue = 0.0


