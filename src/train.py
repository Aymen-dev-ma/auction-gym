# train.py
import numpy as np
import torch.optim as optim
from AuctionAllocation import SecondPrice
from DeconfounderAllocator import DeconfounderAllocator
from GEstimationBidder import GEstimationBidder
from RLAgent import RLAgent
from Models import PyTorchLogisticRegression

def train_policy_network(policy_net, optimizer, states, actions, rewards, gamma=0.99):
    G = 0
    returns = []
    for r in rewards[::-1]:
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    policy_net.train()
    optimizer.zero_grad()
    loss = 0
    for log_prob, G in zip(actions, returns):
        loss += -log_prob * G
    loss.backward()
    optimizer.step()

if __name__ == '__main__':
    rng = np.random.default_rng(0)
    num_runs = 3
    num_iter = 20
    rounds_per_iter = 10000
    num_participants_per_round = 2
    embedding_size = 5
    embedding_var = 1.0
    obs_embedding_size = 4
    num_items = 12

    allocator = DeconfounderAllocator(rng, embedding_size, num_items)
    bidder = GEstimationBidder(rng)
    item_values = np.random.lognormal(0.1, 0.2, num_items)
    policy_net = PyTorchLogisticRegression(n_dim=obs_embedding_size, n_items=num_items)

    agents = [RLAgent(rng, f"Agent {i+1}", num_items, item_values, allocator, bidder, policy_net=policy_net) for i in range(6)]
    auction = Auction(
        rng,
        SecondPrice(),
        agents,
        agent2items={f"Agent {i+1}": np.random.normal(0.0, embedding_var, size=(num_items, embedding_size)) for i in range(6)},
        agents2item_values={f"Agent {i+1}": item_values for i in range(6)},
        max_slots=1,
        embedding_size=embedding_size,
        embedding_var=embedding_var,
        obs_embedding_size=obs_embedding_size,
        num_participants_per_round=num_participants_per_round
    )

    optimizer = optim.Adam(agents[0].policy_net.parameters(), lr=1e-3)

    for iteration in range(num_iter):
        print(f"==== ITERATION {iteration} ====")
        states, actions, rewards = [], [], []

        for _ in range(rounds_per_iter):
            for agent in agents:
                context = np.random.normal(0, 1, obs_embedding_size)
                bid, item = agent.bid(context)
                auction.simulate_opportunity()

                states.append(context)
                actions.append(item)
                rewards.append(agent.net_utility)

        train_policy_network(agents[0].policy_net, optimizer, states, actions, rewards)

        for agent in agents:
            agent.update(iteration)

        for agent in agents:
            agent.clear_utility()
            agent.clear_logs()
