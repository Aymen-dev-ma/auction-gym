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
from Bidder import *  # EmpiricalShadedBidder, TruthfulBidder, PolicyLearningBidderWithCausalInference
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
    # Multi-slot is currently not fully supported.
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


def simulation_run(num_iter, rounds_per_iter, auction):
    agent2net_utility = defaultdict(list)
    agent2gross_utility = defaultdict(list)
    agent2allocation_regret = defaultdict(list)
    agent2estimation_regret = defaultdict(list)
    agent2overbid_regret = defaultdict(list)
    agent2underbid_regret = defaultdict(list)
    agent2CTR_RMSE = defaultdict(list)
    agent2CTR_bias = defaultdict(list)
    agent2gamma = defaultdict(list)
    agent2best_expected_value = defaultdict(list)
    auction_revenue = []

    for i in range(num_iter):
        print(f'==== ITERATION {i} ====')

        for _ in tqdm(range(rounds_per_iter)):
            auction.simulate_opportunity()

        names = [agent.name for agent in auction.agents]
        net_utilities = [agent.net_utility for agent in auction.agents]
        gross_utilities = [agent.gross_utility for agent in auction.agents]

        result = pd.DataFrame({'Name': names, 'Net': net_utilities, 'Gross': gross_utilities})

        print(result)
        print(f'\tAuction revenue: \t {auction.revenue}')

        for agent_id, agent in enumerate(auction.agents):
            agent.update(iteration=i, plot=True, figsize=(10, 6), fontsize=12)

            agent2net_utility[agent.name].append(agent.net_utility)
            agent2gross_utility[agent.name].append(agent.gross_utility)

            agent2allocation_regret[agent.name].append(agent.get_allocation_regret())
            agent2estimation_regret[agent.name].append(agent.get_estimation_regret())
            agent2overbid_regret[agent.name].append(agent.get_overbid_regret())
            agent2underbid_regret[agent.name].append(agent.get_underbid_regret())

            agent2CTR_RMSE[agent.name].append(agent.get_CTR_RMSE())
            agent2CTR_bias[agent.name].append(agent.get_CTR_bias())

            if isinstance(agent.bidder, PolicyLearningBidder) or isinstance(agent.bidder, DoublyRobustBidder):
                agent2gamma[agent.name].append(torch.mean(torch.Tensor(agent.bidder.gammas).detach()).item())
            elif not agent.bidder.truthful:
                agent2gamma[agent.name].append(np.mean([g.detach().numpy() if isinstance(g, torch.Tensor) else g for g in agent.bidder.gammas]))

            best_expected_value = np.mean([opp.best_expected_value for opp in agent.logs])
            agent2best_expected_value[agent.name].append(best_expected_value)

            print('Average Best Value for Agent: ', best_expected_value)
            agent.clear_utility()
            agent.clear_logs()

        auction_revenue.append(auction.revenue)
        auction.clear_revenue()

    return agent2net_utility, agent2gross_utility, auction_revenue


def measure_per_agent2df(run2agent2measure, measure_name):
    df_rows = {'Run': [], 'Agent': [], 'Iteration': [], measure_name: []}
    for run, agent2measure in run2agent2measure.items():
        for agent, measures in agent2measure.items():
            for iteration, measure in enumerate(measures):
                df_rows['Run'].append(run)
                df_rows['Agent'].append(agent)
                df_rows['Iteration'].append(iteration)
                df_rows[measure_name].append(measure)
    return pd.DataFrame(df_rows)


def plot_overlayed_line_plots(run2agent2measure, measure_name):
    df = measure_per_agent2df(run2agent2measure, measure_name)

    fig, axes = plt.subplots(figsize=(10, 6))
    plt.title(f'{measure_name} Over Time', fontsize=14)
    sns.lineplot(data=df, x="Iteration", y=measure_name, hue="Agent", ax=axes)
    plt.xticks(fontsize=12)
    plt.ylabel(f'{measure_name}', fontsize=14)
    plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
    plt.tight_layout()
    plt.savefig(f"results/{measure_name.replace(' ', '_')}_overlayed_line_plot.pdf", bbox_inches='tight')
    plt.show()
    return df


def plot_bar_with_error_bars(run2agent2measure, measure_name):
    df = measure_per_agent2df(run2agent2measure, measure_name)

    summary_df = df.groupby('Agent')[measure_name].agg(['mean', 'std']).reset_index()

    fig, axes = plt.subplots(figsize=(10, 6))
    plt.title(f'{measure_name} Comparison', fontsize=14)
    sns.barplot(data=summary_df, x="Agent", y="mean", yerr=summary_df["std"], ax=axes)
    plt.ylabel(f'{measure_name} (Mean ± Std)', fontsize=14)
    plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
    plt.tight_layout()
    plt.savefig(f"results/{measure_name.replace(' ', '_')}_bar_plot.pdf", bbox_inches='tight')
    plt.show()
    return summary_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to experiment configuration file')
    args = parser.parse_args()

    rng, config, agent_configs, agents2items, agents2item_values, num_runs, max_slots, embedding_size, embedding_var, obs_embedding_size = parse_config(args.config)

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

    for run in range(num_runs):
        agents = instantiate_agents(rng, agent_configs, agents2item_values, agents2items)
        auction, num_iter, rounds_per_iter, output_dir = instantiate_auction(rng, config, agents2items, agents2item_values, agents, max_slots, embedding_size, embedding_var, obs_embedding_size)

        agent2net_utility, agent2gross_utility, auction_revenue = simulation_run(num_iter, rounds_per_iter, auction)

        run2agent2net_utility[run] = agent2net_utility
        run2agent2gross_utility[run] = agent2gross_utility
        run2auction_revenue[run] = auction_revenue

    plot_overlayed_line_plots(run2agent2net_utility, 'Net Utility')
    plot_overlayed_line_plots(run2agent2gross_utility, 'Gross Utility')
    plot_overlayed_line_plots(run2auction_revenue, 'Auction Revenue')

    plot_bar_with_error_bars(run2agent2net_utility, 'Net Utility')
    plot_bar_with_error_bars(run2agent2gross_utility, 'Gross Utility')
    plot_bar_with_error_bars(run2auction_revenue, 'Auction Revenue')
