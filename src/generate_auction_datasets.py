import argparse
import json
import numpy as np
import os
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from Agent import Agent
from AuctionAllocation import *  # FirstPrice, SecondPrice
from Auction import Auction
from Bidder import *  # EmpiricalShadedBidder, TruthfulBidder
from BidderAllocation import *  # LogisticTSAllocator, OracleAllocator


def parse_kwargs(kwargs):
    parsed = ','.join([f'{key}={value}' for key, value in kwargs.items()])
    return ',' + parsed if parsed else ''


def parse_config(path):
    with open(path) as f:
        config = json.load(f)

    # Set up Random Number Generator
    rng = np.random.default_rng(config['random_seed'])
    np.random.seed(config['random_seed'])

    # Number of runs
    num_runs = config['num_runs'] if 'num_runs' in config.keys() else 1

    # Max. number of slots in every auction round
    max_slots = 1

    # Technical parameters for distribution of latent embeddings
    embedding_size = config['embedding_size']
    embedding_var = config['embedding_var']
    obs_embedding_size = config['obs_embedding_size']

    # Expand agent-config if there are multiple copies
    agent_configs = []
    num_agents = 0
    for agent_config in config['agents']:
        if 'num_copies' in agent_config.keys():
            for i in range(1, agent_config['num_copies'] + 1):
                agent_config_copy = deepcopy(agent_config)
                agent_config_copy['name'] += f' {num_agents + 1}'
                agent_configs.append(agent_config_copy)
                num_agents += 1
        else:
            agent_configs.append(agent_config)
            num_agents += 1

    # First sample item catalog (so it is consistent over different configs with the same seed)
    # Agent : (item_embedding, item_value)
    agents2items = {
        agent_config['name']: rng.normal(0.0, embedding_var, size=(agent_config['num_items'], embedding_size))
        for agent_config in agent_configs
    }

    agents2item_values = {
        agent_config['name']: rng.lognormal(0.1, 0.2, agent_config['num_items'])
        for agent_config in agent_configs
    }

    # Add intercepts to embeddings (Uniformly in [-4.5, -1.5], this gives nicer distributions for P(click))
    for agent, items in agents2items.items():
        agents2items[agent] = np.hstack((items, - 3.0 - 1.0 * rng.random((items.shape[0], 1))))

    return rng, config, agent_configs, agents2items, agents2item_values, num_runs, max_slots, embedding_size, embedding_var, obs_embedding_size


def instantiate_agents(rng, agent_configs, agents2item_values, agents2items):
    # Store agents to be re-instantiated in subsequent runs
    # Set up agents
    agents = [
        Agent(rng=rng,
              name=agent_config['name'],
              num_items=agent_config['num_items'],
              item_values=agents2item_values[agent_config['name']],
              allocator=eval(f"{agent_config['allocator']['type']}(rng=rng{parse_kwargs(agent_config['allocator']['kwargs'])})"),
              bidder=eval(f"{agent_config['bidder']['type']}(rng=rng{parse_kwargs(agent_config['bidder']['kwargs'])})"),
              memory=(0 if 'memory' not in agent_config.keys() else agent_config['memory']))
        for agent_config in agent_configs
    ]

    for agent in agents:
        if isinstance(agent.allocator, OracleAllocator):
            agent.allocator.update_item_embeddings(agents2items[agent.name])

    return agents


def instantiate_auction(rng, config, agents2items, agents2item_values, agents, max_slots, embedding_size, embedding_var, obs_embedding_size):
    return (Auction(rng,
                    eval(f"{config['allocation']}()"),
                    agents,
                    agents2items,
                    agents2item_values,
                    max_slots,
                    embedding_size,
                    embedding_var,
                    obs_embedding_size,
                    config['num_participants_per_round']),
            config['num_iter'], config['rounds_per_iter'], config['output_dir'])


def save_data(auction, output_dir, prefix):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Print debug info
    print(f"Saving data to {output_dir}/{prefix}_data.npz")

    # Extract and prepare data to be saved
    data = {
        'contexts': np.array([opp.context for agent in auction.agents for opp in agent.logs]),
        'items': np.array([opp.item for agent in auction.agents for opp in agent.logs]),
        'values': np.array([opp.value for agent in auction.agents for opp in agent.logs]),
        'bids': np.array([opp.bid for agent in auction.agents for opp in agent.logs]),
        'prices': np.array([opp.price for agent in auction.agents for opp in agent.logs]),
        'outcomes': np.array([opp.outcome for agent in auction.agents for opp in agent.logs]),
        'estimated_CTRs': np.array([opp.estimated_CTR for agent in auction.agents for opp in agent.logs]),
        'true_CTRs': np.array([opp.true_CTR for agent in auction.agents for opp in agent.logs])
    }

    # Split data into training, validation, and testing sets
    x_train, x_temp, a_train, a_temp, r_train, r_temp, u_train, u_temp = train_test_split(
        data['contexts'], data['items'], data['outcomes'], data['true_CTRs'], test_size=0.4, random_state=42)
    
    x_val, x_test, a_val, a_test, r_val, r_test, u_val, u_test = train_test_split(
        x_temp, a_temp, r_temp, u_temp, test_size=0.5, random_state=42)

    # Save datasets
    np.savez_compressed(os.path.join(output_dir, 'training_data.npz'), 
                        x_seq=x_train, a_seq=a_train, r_seq=r_train, u_seq=u_train)
    
    np.savez_compressed(os.path.join(output_dir, 'validation_data.npz'), 
                        x_seq=x_val, a_seq=a_val, r_seq=r_val, u_seq=u_val)
    
    np.savez_compressed(os.path.join(output_dir, 'testing_data.npz'), 
                        x_seq=x_test, a_seq=a_test, r_seq=r_test, u_seq=u_test)

    print(f"Data saved successfully to {output_dir}/{prefix}_data.npz")


def simulation_run():
    for i in range(num_iter):
        print(f'==== ITERATION {i} ====')

        for _ in tqdm(range(rounds_per_iter)):
            auction.simulate_opportunity()

        save_data(auction, output_dir, f'iteration_{i}')

        # Optionally reset auction for next iteration
        auction.clear_revenue()
        for agent in auction.agents:
            agent.clear_utility()
            agent.clear_logs()


if __name__ == '__main__':
    # Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to experiment configuration file')
    args = parser.parse_args()

    # Parse configuration file
    rng, config, agent_configs, agents2items, agents2item_values, num_runs, max_slots, embedding_size, embedding_var, obs_embedding_size = parse_config(args.config)

    # Repeated runs
    for run in range(num_runs):
        # Reinstantiate agents and auction per run
        agents = instantiate_agents(rng, agent_configs, agents2item_values, agents2items)
        auction, num_iter, rounds_per_iter, output_dir = instantiate_auction(rng, config, agents2items, agents2item_values, agents, max_slots, embedding_size, embedding_var, obs_embedding_size)

        # Run simulation and save data
        simulation_run()
