import mlflow
import mlflow.pytorch
import numpy as np
import torch
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from Agent import Agent
from AuctionAllocation import FirstPrice, SecondPrice
from Auction import Auction
from Bidder import EmpiricalShadedBidder, TruthfulBidder, PolicyLearningBidderWithCausalInference
from BidderAllocation import LogisticTSAllocator, OracleAllocator

def parse_kwargs(kwargs):
    parsed = ','.join([f'{key}={value}' for key, value in kwargs.items()])
    return ',' + parsed if parsed else ''

def parse_config(path):
    with open(path) as f:
        config = json.load(f)
    rng = np.random.default_rng(config['random_seed'])
    np.random.seed(config['random_seed'])
    num_runs = config['num_runs'] if 'num_runs' in config.keys() else 1
    max_slots = 1
    embedding_size = config['embedding_size']
    embedding_var = config['embedding_var']
    obs_embedding_size = config['obs_embedding_size']
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
    agents2items = {
        agent_config['name']: rng.normal(0.0, embedding_var, size=(agent_config['num_items'], embedding_size))
        for agent_config in agent_configs
    }
    agents2item_values = {
        agent_config['name']: rng.lognormal(0.1, 0.2, agent_config['num_items'])
        for agent_config in agent_configs
    }
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

def log_metrics_and_params(agent, iteration, metrics):
    with mlflow.start_run(nested=True):
        mlflow.log_param("agent_name", agent.name)
        mlflow.log_param("iteration", iteration)
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value, step=iteration)

def simulation_run(num_iter, rounds_per_iter, auction):
    agent2net_utility = defaultdict(list)
    agent2gross_utility = defaultdict(list)
    auction_revenue = []

    for i in range(num_iter):
        print(f'==== ITERATION {i} ====')
        for _ in tqdm(range(rounds_per_iter)):
            auction.simulate_opportunity()
        for agent_id, agent in enumerate(auction.agents):
            agent.update(iteration=i, plot=True, figsize=(10, 6), fontsize=12)
            net_utility = agent.net_utility
            gross_utility = agent.gross_utility
            agent2net_utility[agent.name].append(net_utility)
            agent2gross_utility[agent.name].append(gross_utility)
            metrics = {
                "net_utility": net_utility,
                "gross_utility": gross_utility,
                "allocation_regret": agent.get_allocation_regret(),
                "estimation_regret": agent.get_estimation_regret(),
                "overbid_regret": agent.get_overbid_regret(),
                "underbid_regret": agent.get_underbid_regret(),
                "CTR_RMSE": agent.get_CTR_RMSE(),
                "CTR_bias": agent.get_CTR_bias(),
            }
            log_metrics_and_params(agent, i, metrics)
            agent.clear_utility()
            agent.clear_logs()
        auction_revenue.append(auction.revenue)
        auction.clear_revenue()
    return agent2net_utility, agent2gross_utility, auction_revenue

if __name__ == '__main__':
    import json
    import os
    from copy import deepcopy

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to experiment configuration file')
    args = parser.parse_args()

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Policy_Optimization_Causal_Inference")

    rng, config, agent_configs, agents2items, agents2item_values, num_runs, max_slots, embedding_size, embedding_var, obs_embedding_size = parse_config(args.config)

    run2agent2net_utility = {}
    run2agent2gross_utility = {}
    run2auction_revenue = {}

    with mlflow.start_run():
        for run in range(num_runs):
            agents = instantiate_agents(rng, agent_configs, agents2item_values, agents2items)
            auction, num_iter, rounds_per_iter, output_dir = instantiate_auction(rng, config, agents2items, agents2item_values, agents, max_slots, embedding_size, embedding_var, obs_embedding_size)
            agent2net_utility, agent2gross_utility, auction_revenue = simulation_run(num_iter, rounds_per_iter, auction)
            run_metrics = {
                "final_net_utility": sum(sum(agent2net_utility.values(), [])),
                "final_gross_utility": sum(sum(agent2gross_utility.values(), [])),
                "total_auction_revenue": sum(auction_revenue),
            }
            mlflow.log_metrics(run_metrics)
