import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from Auction import Auction
from Agent import Agent
from AuctionAllocation import *  # Allocation Mechanism classes
from Bidder import *  # Bidders
from BidderAllocation import *  # Allocators

# -------------------- Parse the Configuration File --------------------

def parse_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Set up random number generator
    rng = np.random.default_rng(config['random_seed'])
    np.random.seed(config['random_seed'])

    # Read agent configurations
    agent_configs = config['agents']
    
    return rng, config, agent_configs

# -------------------- Instantiate Auction and Agents --------------------

def instantiate_agents_and_auction(rng, config):
    # Initialize agents and their respective items
    agents2items = {
        agent['name']: rng.normal(0.0, config['embedding_var'], size=(agent['num_items'], config['embedding_size']))
        for agent in config['agents']
    }
    
    agents2item_values = {
        agent['name']: rng.lognormal(0.1, 0.2, agent['num_items'])
        for agent in config['agents']
    }

    # Create agents
    agents = []
    for agent_config in config['agents']:
        allocator = eval(f"{agent_config['allocator']['type']}(rng=rng)")
        bidder = eval(f"{agent_config['bidder']['type']}(rng=rng, **agent_config['bidder']['kwargs'])")
        agent = Agent(rng=rng, name=agent_config['name'], num_items=agent_config['num_items'], 
                      item_values=agents2item_values[agent_config['name']], allocator=allocator, bidder=bidder)
        agents.append(agent)

    # Create auction with multiple agents
    auction = Auction(rng, eval(f"{config['allocation']}()"), agents, agents2items, agents2item_values, 
                      config['max_slots'], config['embedding_size'], config['embedding_var'], 
                      config['obs_embedding_size'], config['num_participants_per_round'])

    return auction, agents

# -------------------- Simulate Auction Rounds and Collect Data --------------------

def simulate_auction_and_collect_data(auction, num_rounds):
    data = {
        "bidding_preference": [],    # Y: Latent factor - agent's bidding preference
        "item_latent_value": [],     # Y: Latent factor - latent value perceived by the agent
        "market_influence": [],      # Y: Latent factor - hidden market influence
        "item_features": [],         # X: Observable item features
        "market_context": [],        # X: Observable market conditions
        "agent_history_features": [] # X: Observable agent history features
    }
    
    for _ in tqdm(range(num_rounds), desc="Simulating auction rounds"):
        auction.simulate_opportunity()

        for agent in auction.agents:
            for log in agent.logs:
                # Log latent factors (Y)
                bidding_preference = agent.name  # Latent: agent's bidding preference
                item_latent_value = log.value    # Latent: perceived value of the item by the agent
                market_influence = log.context[-1]  # Latent: hidden market conditions
                
                # Log observable states (X)
                item_features = auction.agent2items[agent.name]  # Observable: item features
                market_context = log.obs_context  # Observable: market context (e.g., competitor bids)
                agent_history_features = np.mean(agent.logs[-5:], axis=0)  # Observable: past bids, average bid
                
                # Add the data to the dataset
                data["bidding_preference"].append(bidding_preference)
                data["item_latent_value"].append(item_latent_value)
                data["market_influence"].append(market_influence)
                data["item_features"].append(item_features)
                data["market_context"].append(market_context)
                data["agent_history_features"].append(agent_history_features)

    return pd.DataFrame(data)

# -------------------- Main Script to Generate Dataset --------------------

def main(config_path, output_path, num_rounds):
    # Parse the configuration
    rng, config, agent_configs = parse_config(config_path)

    # Instantiate auction and agents
    auction, agents = instantiate_agents_and_auction(rng, config)

    # Simulate auction and collect dataset
    dataset = simulate_auction_and_collect_data(auction, num_rounds)

    # Save the dataset to a CSV file
    dataset.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description="Auction Simulator for Dataset Generation")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration JSON file")
    parser.add_argument('--output', type=str, required=True, help="Path to save the generated dataset CSV")
    parser.add_argument('--rounds', type=int, default=10000, help="Number of auction rounds to simulate")

    args = parser.parse_args()

    # Run the main dataset generation process
    main(args.config, args.output, args.rounds)
