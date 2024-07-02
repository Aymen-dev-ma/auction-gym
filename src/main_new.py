# main_new.py

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm

from Agent import Agent
from AuctionAllocation import *  # FirstPrice, SecondPrice
from Auction import Auction
from Bidder import *  # EmpiricalShadedBidder, TruthfulBidder
from BidderAllocation import *  # LogisticTSAllocator, OracleAllocator
from policy_learning_bidder_with_causal_inference import PolicyLearningBidderWithCausalInference

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

def simulation_run():
    global num_iter, rounds_per_iter, auction, agent2net_utility, agent2gross_utility, agent2allocation_regret, agent2estimation_regret, agent2overbid_regret, agent2underbid_regret, agent2CTR_RMSE, agent2CTR_bias, agent2gamma, agent2best_expected_value, auction_revenue, FIGSIZE, FONTSIZE

    iteration_results = []

    for i in range(num_iter):
        print(f'==== ITERATION {i} ====')

        for _ in tqdm(range(rounds_per_iter)):
            auction.simulate_opportunity()

        names = [agent.name for agent in auction.agents]
        net_utilities = [agent.net_utility for agent in auction.agents]
        gross_utilities = [agent.gross_utility for agent in auction.agents]

        result = pd.DataFrame({'Name': names, 'Net Utility': net_utilities, 'Gross Utility': gross_utilities, 'Iteration': i})
        iteration_results.append(result)

        print(result)
        print(f'\tAuction revenue: \t {auction.revenue}')

        for agent_id, agent in enumerate(auction.agents):
            agent.update(iteration=i, plot=True, figsize=FIGSIZE, fontsize=FONTSIZE)

            agent2net_utility[agent.name].append(agent.net_utility)
            agent2gross_utility[agent.name].append(agent.gross_utility)

            agent2allocation_regret[agent.name].append(agent.get_allocation_regret())
            agent2estimation_regret[agent.name].append(agent.get_estimation_regret())
            agent2overbid_regret[agent.name].append(agent.get_overbid_regret())
            agent2underbid_regret[agent.name].append(agent.get_underbid_regret())

            agent2CTR_RMSE[agent.name].append(agent.get_CTR_RMSE())
            agent2CTR_bias[agent.name].append(agent.get_CTR_bias())

            if isinstance(agent.bidder, PolicyLearningBidder) or isinstance(agent.bidder, PolicyLearningBidderWithCausalInference):
                agent2gamma[agent.name].append(torch.mean(torch.Tensor(agent.bidder.gammas)).detach().item())
            elif not agent.bidder.truthful:
                agent2gamma[agent.name].append(np.mean(agent.bidder.gammas))

            best_expected_value = np.mean([opp.best_expected_value for opp in agent.logs])
            agent2best_expected_value[agent.name].append(best_expected_value)

            print('Average Best Value for Agent: ', best_expected_value)
            agent.clear_utility()
            agent.clear_logs()

        auction_revenue.append(auction.revenue)
        auction.clear_revenue()

    return iteration_results

def run_simulation(config_file):
    rng, config, agent_configs, agents2items, agents2item_values, num_runs, max_slots, embedding_size, embedding_var, obs_embedding_size = parse_config(config_file)
    results = []
    for run in range(num_runs):
        agents = instantiate_agents(rng, agent_configs, agents2item_values, agents2items)
        auction, num_iter, rounds_per_iter, output_dir = instantiate_auction(rng, config, agents2items, agents2item_values, agents, max_slots, embedding_size, embedding_var, obs_embedding_size)
        result = simulation_run()
        results.append(pd.concat(result))
    return results

def compare_results(baseline_results, causal_results):
    baseline_df = pd.concat(baseline_results)
    causal_df = pd.concat(causal_results)

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.lineplot(data=baseline_df, x='Iteration', y='Net Utility', hue='Name', ax=ax, label='Baseline')
    sns.lineplot(data=causal_df, x='Iteration', y='Net Utility', hue='Name', ax=ax, label='Causal Inference', linestyle='--')

    ax.set_title('Comparison of Net Utility Over Time')
    ax.set_ylabel('Net Utility')
    ax.set_xlabel('Iteration')
    ax.legend(title='Approach')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('comparison_net_utility.png')
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.lineplot(data=baseline_df, x='Iteration', y='Gross Utility', hue='Name', ax=ax, label='Baseline')
    sns.lineplot(data=causal_df, x='Iteration', y='Gross Utility', hue='Name', ax=ax, label='Causal Inference', linestyle='--')

    ax.set_title('Comparison of Gross Utility Over Time')
    ax.set_ylabel('Gross Utility')
    ax.set_xlabel('Iteration')
    ax.legend(title='Approach')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('comparison_gross_utility.png')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline_config', type=str, help='Path to baseline configuration file')
    parser.add_argument('causal_config', type=str, help='Path to causal inference configuration file')
    args = parser.parse_args()

    FIGSIZE = (8, 5)
    FONTSIZE = 14

    # Placeholders for summary statistics over all runs
    run2agent2net_utility = {}
    run2agent2gross_utility = {}
    run2agent2allocation_regret = {}
    run2agent2estimation_regret = {}
    run2agent2overbid_regret = {}
    run2agent2underbid_regret = {}
    run2agent2best_expected_value = {}

    run2agent2CTR_RMSE = {}
    run2agent2CTR_bias = {}
    run2agent2gamma = {}

    run2auction_revenue = {}

    # Run baseline
    baseline_results = run_simulation(args.baseline_config)

    # Run causal inference
    causal_results = run_simulation(args.causal_config)

    # Compare results
    compare_results(baseline_results, causal_results)
