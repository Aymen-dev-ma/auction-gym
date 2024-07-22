import json
import numpy as np
from copy import deepcopy

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
    agents2items = {agent_config['name']: rng.normal(0.0, embedding_var, size=(agent_config['num_items'], embedding_size)) for agent_config in agent_configs}
    agents2item_values = {agent_config['name']: np.ones(agent_config['num_items']) for agent_config in agent_configs}
    return rng, config, agent_configs, agents2items, agents2item_values, num_runs, max_slots, embedding_size, embedding_var, obs_embedding_size
