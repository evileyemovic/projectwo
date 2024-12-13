import os
import argparse
import time
import torch
import random
import numpy as np

from enum import Enum

from copy import deepcopy

from rlcard.games.nolimitholdem import Game

from rlcard.games.base import Card

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)
from rlcard.agents import NolimitholdemHumanAgent

class Stage(Enum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    END_HIDDEN = 4
    SHOWDOWN = 5


def traincs(args):


    # Check whether gpu is available
    device = get_device()
        
    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(
        args.env,
        config={
            'seed': int(time.time()),
        }
    )

    # Initialize the agent and use random agents as opponents
    if args.algorithm == 'dqn':
        from rlcard.agents import DQNAgent_CS
        agent = DQNAgent_CS(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            mlp_layers=[64,64],
            device=device,
        )

    load_path = os.path.join(args.log_dircs, 'model.pth')
    agent = torch.load(load_path)


    hagent = NolimitholdemHumanAgent(num_actions=env.num_actions)
    agents = [agent, hagent]
    env.set_agents(agents)


    while(1):
        state, player_id = env.reset()
        print("start game!!!!")
        print("dealer is : %d" % env.game.dealer_id)
   

        while not env.is_over():
            if player_id == 0:
                action, info = env.agents[player_id].eval_step(state)
                next_state, next_player_id = env.step(action, env.agents[player_id].use_raw)
                state = next_state
                player_id = next_player_id
            else:
                action = env.agents[player_id].step(state)
                next_state, next_player_id = env.step(action)
                state = next_state
                player_id = next_player_id

        for player_id in range(env.num_players):
            state = env.get_state(player_id)
            print(state['raw_obs']['hand'])
            # Payoffs
        payoffs = env.get_payoffs()
        print(payoffs)

def trainma(args):


    # Check whether gpu is available
    device = get_device()
        
    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(
        args.env,
        config={
            'seed': int(time.time()),
        }
    )

    # Initialize the agent and use random agents as opponents
    if args.algorithm == 'dqn':
        from rlcard.agents import DQNAgent_MA
        agent = DQNAgent_MA(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            mlp_layers=[64,64],
            device=device,
        )

    load_path = os.path.join(args.log_dirma, 'model.pth')
    agent = torch.load(load_path)


    hagent = NolimitholdemHumanAgent(num_actions=env.num_actions)
    agents = [agent, hagent]
    env.set_agents(agents)


    while(1):
        state, player_id = env.reset()
        print("start game!!!!")
        print("dealer is : %d" % env.game.dealer_id)
   

        while not env.is_over():
            if player_id == 0:
                action, info = env.agents[player_id].eval_step(state)
                next_state, next_player_id = env.step(action, env.agents[player_id].use_raw)
                state = next_state
                player_id = next_player_id
            else:
                action = env.agents[player_id].step(state)
                next_state, next_player_id = env.step(action)
                state = next_state
                player_id = next_player_id

        for player_id in range(env.num_players):
            state = env.get_state(player_id)
            print(state['raw_obs']['hand'])
            # Payoffs
        payoffs = env.get_payoffs()
        print(payoffs)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN/NFSP example in RLCard")
    parser.add_argument(
        '--env',
        type=str,
        default='no-limit-holdem',
        choices=[
            'blackjack',
            'leduc-holdem',
            'limit-holdem',
            'doudizhu',
            'mahjong',
            'no-limit-holdem',
            'uno',
            'gin-rummy',
            'bridge',
        ],
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        default='dqn',
        choices=[
            'dqn',
            'nfsp',
        ],
    )
    parser.add_argument(
        '--cuda',
        type=str,
        default='',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=5000,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=2000,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/Advance4/',
    )
    parser.add_argument(
        '--log_dircs',
        type=str,
        default='experiments/Advance_CS4/',
    )
    parser.add_argument(
        '--log_dirma',
        type=str,
        default='experiments/Advance_MA4/',
    )


    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    traincs(args)