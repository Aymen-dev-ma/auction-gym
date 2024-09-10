from AuctionAllocation import AllocationMechanism
from Bidder import Bidder

import numpy as np
import csv
from BidderAllocation import OracleAllocator
from Models import sigmoid
class Auction:
    ''' Base class for auctions with data logging '''
    
    def __init__(self, rng, allocation, agents, agent2items, agents2item_values, max_slots, embedding_size, embedding_var, obs_embedding_size, num_participants_per_round, log_file='auction_dataset.csv'):
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

        # Open a log file to store the auction dataset
        self.log_file = open(log_file, mode='w', newline='')
        self.log_writer = csv.writer(self.log_file)
        # Write the header of the CSV file
        self.log_writer.writerow(['context1', 'context2', 'context3', 'context4', 'true_CTR', 'bid', 'outcome'])

    def simulate_opportunity(self):
        # Sample the number of slots uniformly between [1, max_slots]
        num_slots = self.rng.integers(1, self.max_slots + 1)

        # Sample a true context vector
        true_context = np.concatenate((self.rng.normal(0, self.embedding_var, size=self.embedding_size), [1.0]))

        # Mask true context into observable context
        obs_context = np.concatenate((true_context[:self.obs_embedding_size], [1.0]))

        # At this point, the auctioneer solicits bids from
        # the list of bidders that might want to compete.
        bids = []
        CTRs = []
        participating_agents_idx = self.rng.choice(len(self.agents), self.num_participants_per_round, replace=False)
        participating_agents = [self.agents[idx] for idx in participating_agents_idx]

        for agent in participating_agents:
            # Get the bid and the allocated item
            if isinstance(agent.allocator, OracleAllocator):
                bid, item = agent.bid(true_context)
            else:
                bid, item = agent.bid(obs_context)

            bids.append(bid)
            
            # Compute the true CTRs for items in this agent's catalogue
            true_CTR = sigmoid(true_context @ self.agent2items[agent.name].T)
            best_CTR = np.max(true_CTR * self.agents2item_values[agent.name])
            agent.logs[-1].set_true_CTR(best_CTR, true_CTR[item])
            CTRs.append(true_CTR[item])

            # Log context, CTR, bid, and outcome
            context = obs_context[:self.obs_embedding_size]
            self.log_writer.writerow(list(context) + [best_CTR, agent.logs[-1].bid, agent.logs[-1].outcome])

        bids = np.array(bids)
        CTRs = np.array(CTRs)

        # Now we have bids, we need to allocate slots
        winners, prices, second_prices = self.allocation.allocate(bids, num_slots)

        # Bidders only obtain value when they get their outcome
        outcomes = self.rng.binomial(1, CTRs[winners])

        # Let bidders know what they're being charged for
        for slot_id, (winner, price, second_price, outcome) in enumerate(zip(winners, prices, second_prices, outcomes)):
            for agent_id, agent in enumerate(participating_agents):
                if agent_id == winner:
                    agent.charge(price, second_price, bool(outcome))
                else:
                    agent.set_price(price)
            self.revenue += price

    def clear_revenue(self):
        self.revenue = 0.0

    def close_log(self):
        # Close the log file when the simulation is done
        self.log_file.close()
