import numpy as np
from Auction import Auction
from AuctionAllocation import FirstPrice
from Agent import Agent
from Bidder import TruthfulBidder
from BidderAllocation import OracleAllocator

def flatten_context(context):
    """Flattens the context vector to mimic the flattened MNIST images."""
    return context.flatten()

class AuctionWithLatents(Auction):
    def __init__(self, *args, **kwargs):
        super(AuctionWithLatents, self).__init__(*args, **kwargs)
        self.dataset = {'x_seq': [], 'a_seq': [], 'r_seq': [], 'mask_seq': [], 'u_seq': []}

    def simulate_opportunity(self):
        latent_market_condition = self.rng.normal(0, 1, size=1)  # Hidden confounder

        num_slots = self.rng.integers(1, self.max_slots + 1)
        true_context = np.concatenate((self.rng.normal(0, self.embedding_var, size=self.embedding_size), [1.0]))
        obs_context = np.concatenate((true_context[:self.obs_embedding_size], [1.0]))

        bids = []
        CTRs = []
        participating_agents_idx = self.rng.choice(len(self.agents), self.num_participants_per_round, replace=False)
        participating_agents = [self.agents[idx] for idx in participating_agents_idx]

        for agent in participating_agents:
            bid, item = agent.bid(obs_context)
            bids.append(bid)
            true_CTR = 1 / (1 + np.exp(-true_context @ self.agent2items[agent.name].T))
            agent.logs[-1].set_true_CTR(np.max(true_CTR * self.agents2item_values[agent.name]), true_CTR[item])
            CTRs.append(true_CTR[item])

        bids = np.array(bids)
        CTRs = np.array(CTRs)

        winners, prices, second_prices = self.allocation.allocate(bids, num_slots)
        outcomes = self.rng.binomial(1, CTRs[winners])

        for slot_id, (winner, price, second_price, outcome) in enumerate(zip(winners, prices, second_prices, outcomes)):
            agent = participating_agents[winner]
            agent.charge(price, second_price, bool(outcome))

            flattened_context = flatten_context(obs_context)
            self.dataset['x_seq'].append(flattened_context)
            self.dataset['a_seq'].append([bids[winner]])
            self.dataset['r_seq'].append([outcome])
            self.dataset['mask_seq'].append([1.0])
            self.dataset['u_seq'].append(latent_market_condition)

    def save_dataset(self, filename_prefix, num_train, num_val_test, nsteps, x_dim):
        self.dataset = {k: np.array(v) for k, v in self.dataset.items()}
        
        # Reshape to match expected dimensions
        self.dataset['x_seq'] = self.dataset['x_seq'].reshape(-1, nsteps, x_dim)
        self.dataset['a_seq'] = np.array(self.dataset['a_seq']).reshape(-1, nsteps, 1)
        self.dataset['r_seq'] = np.array(self.dataset['r_seq']).reshape(-1, nsteps, 1)
        self.dataset['mask_seq'] = np.array(self.dataset['mask_seq']).reshape(-1, nsteps, 1)
        self.dataset['u_seq'] = np.array(self.dataset['u_seq']).reshape(-1, nsteps, 1)

        # Split into training, validation, and test sets
        x_train = self.dataset['x_seq'][:num_train]
        a_train = self.dataset['a_seq'][:num_train]
        r_train = self.dataset['r_seq'][:num_train]
        mask_train = self.dataset['mask_seq'][:num_train]
        rich_train = self.dataset['u_seq'][:num_train]

        x_val_test = self.dataset['x_seq'][num_train:]
        a_val_test = self.dataset['a_seq'][num_train:]
        r_val_test = self.dataset['r_seq'][num_train:]
        mask_val_test = self.dataset['mask_seq'][num_train:]
        rich_val_test = self.dataset['u_seq'][num_train:]

        x_validation = x_val_test[:num_val_test]
        a_validation = a_val_test[:num_val_test]
        r_validation = r_val_test[:num_val_test]
        mask_validation = mask_val_test[:num_val_test]
        rich_validation = rich_val_test[:num_val_test]

        x_test = x_val_test[num_val_test:]
        a_test = a_val_test[num_val_test:]
        r_test = r_val_test[num_val_test:]
        mask_test = mask_val_test[num_val_test:]
        rich_test = rich_val_test[num_val_test:]

        # Save datasets as npz files
        np.savez(f'{filename_prefix}_training_data.npz', x_train=x_train, a_train=a_train, r_train=r_train, mask_train=mask_train, rich_train=rich_train)
        np.savez(f'{filename_prefix}_validation_data.npz', x_validation=x_validation, a_validation=a_validation, r_validation=r_validation, mask_validation=mask_validation, rich_validation=rich_validation)
        np.savez(f'{filename_prefix}_testing_data.npz', x_test=x_test, a_test=a_test, r_test=r_test, mask_test=mask_test, rich_test=rich_test)

        print(f"Dataset saved with prefix {filename_prefix}.")

def instantiate_agents_and_auction(rng, config):
    agents = [
        Agent(rng=rng,
              name=agent_config['name'],
              num_items=agent_config['num_items'],
              item_values=config['agents2item_values'][agent_config['name']],
              allocator=OracleAllocator(rng),
              bidder=TruthfulBidder(rng))
        for agent_config in config['agent_configs']
    ]

    for agent in agents:
        agent.allocator.update_item_embeddings(config['agents2items'][agent.name])

    auction = AuctionWithLatents(rng=rng,
                                 allocation=FirstPrice(),
                                 agents=agents,
                                 agent2items=config['agents2items'],
                                 agents2item_values=config['agents2item_values'],
                                 max_slots=config['max_slots'],
                                 embedding_size=config['embedding_size'],
                                 embedding_var=config['embedding_var'],
                                 obs_embedding_size=config['obs_embedding_size'],
                                 num_participants_per_round=config['num_participants_per_round'])

    return auction

def main():
    rng = np.random.default_rng(42)

    config = {
        'embedding_size': 5,
        'embedding_var': 0.1,
        'obs_embedding_size': 3,
        'num_participants_per_round': 3,
        'max_slots': 1,
        'agents2items': {
            'Agent 1': rng.normal(0.0, 0.1, size=(10, 5)),
            'Agent 2': rng.normal(0.0, 0.1, size=(10, 5)),
            'Agent 3': rng.normal(0.0, 0.1, size=(10, 5))
        },
        'agents2item_values': {
            'Agent 1': rng.lognormal(0.1, 0.2, 10),
            'Agent 2': rng.lognormal(0.1, 0.2, 10),
            'Agent 3': rng.lognormal(0.1, 0.2, 10)
        },
        'agent_configs': [{'name': 'Agent 1', 'num_items': 10},
                          {'name': 'Agent 2', 'num_items': 10},
                          {'name': 'Agent 3', 'num_items': 10}]
    }

    # Parameters matching the MNIST dataset
    num_train = 60000
    num_val_test = 10000
    nsteps = 5
    x_dim = 784  # flattened image size (28x28)

    auction = instantiate_agents_and_auction(rng, config)

    num_rounds = num_train + 2 * num_val_test  # Train + Val + Test
    for _ in range(num_rounds):
        auction.simulate_opportunity()

    auction.save_dataset('auction_deconfounding_dataset', num_train, num_val_test, nsteps, x_dim)

if __name__ == "__main__":
    main()
