import itertools
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

import os
import argparse
import time
import torch
import random

from enum import Enum

from copy import deepcopy

from rlcard.games.nolimitholdem import Game
from rlcard.games.nolimitholdem.round import Action

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

# 1. 모든 가능한 핸드 생성
def generate_all_hands():
    """
    모든 가능한 핸드 조합(52장의 카드 중 중복 제거 포함)을 생성.
    - KsAd와 AdKs는 하나로 간주.
    - AhAh와 같은 불가능한 핸드는 필터링.
    """
    suits = ['S', 'H', 'D', 'C']
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    deck = [rank + suit for rank in ranks for suit in suits]

    hands = []
    for card1, card2 in itertools.combinations(deck, 2):
        rank1, suit1 = card1[:-1], card1[-1]
        rank2, suit2 = card2[:-1], card2[-1]

        # 같은 카드는 허용하지 않음
        if card1 == card2:
            continue

        # 순서 정규화: 높은 랭크가 첫 번째로 오도록
        if rank1 > rank2 or (rank1 == rank2 and suit1 > suit2):
            hands.append((card2, card1))
        else:
            hands.append((card1, card2))

    return hands

# 2. 핸드를 그룹화 (AA, AKs, AKo 등)
def group_hands_by_type(hands):
    """
    핸드를 그룹화 (예: AA, AKs, AKo).
    """
    grouped_hands = {}
    suits = ['S', 'H', 'D', 'C']

    for hand in hands:
        rank1, suit1 = hand[0][:-1], hand[0][-1]
        rank2, suit2 = hand[1][:-1], hand[1][-1]

        if rank1 == rank2:  # Pair (AA, KK)
            key = rank1 + rank2
        elif suit1 == suit2:  # Suited (AKs)
            key = rank1 + rank2 + 's' if rank1 > rank2 else rank2 + rank1 + 's'
        else:  # Offsuit (AKo)
            key = rank1 + rank2 + 'o' if rank1 > rank2 else rank2 + rank1 + 'o'

        if key not in grouped_hands:
            grouped_hands[key] = []
        grouped_hands[key].append(hand)

    return grouped_hands

# 3. 특정 상황에서 에이전트의 행동 평가
def evaluate_hands(agent, public_cards, action_record, env):
    """
    특정 상황에서 모든 핸드에 대한 에이전트의 행동 평가.
    """
    hands = generate_all_hands()
    grouped_hands = group_hands_by_type(hands)
    hand_actions = {}

    for group, hands in grouped_hands.items():
        actions = []
        for hand in hands: 
            state = {'hand': [], 'public_cards': [], 'all_chips': [4, 4], 'my_chips': 4, 'legal_actions': [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT, Action.RAISE_POT, Action.ALL_IN], 'stakes': [96, 96], 'current_player': 0, 'pot': 8, 'stage': Stage.FLOP}
            state['hand'] = [hand[1][1]+hand[1][0], hand[0][1]+hand[0][0]]
            state['public_cards'] = public_cards
            flag = True
            for i in state['public_cards']:
                if i == hand[1][1]+hand[1][0] or i == hand[0][1]+hand[0][0]:
                    flag = False
                    continue
            if not flag:
                continue
            env.action_recorder  = action_record
            state = env._extract_state(state)
            action, _ = agent.eval_step(state)
            actions.append(action)
        #most_common_action = Counter(actions).most_common(1)[0][0]
        #hand_actions[group] = most_common_action

        if actions:  # 액션이 있을 때만 계산
            avg_action_value = sum(actions) / len(actions)
        else:
            avg_action_value = 1  # 액션이 없을 경우 기본값

        hand_actions[group] = avg_action_value

    return hand_actions

# 4. 시각화
def plot_hand_range(hand_actions):
    """
    핸드 레인지를 시각화. 배열:
    - AA: 왼쪽 위, 22: 오른쪽 아래
    - 수딧 핸드: 오른쪽 위, 오프수딧 핸드: 왼쪽 아래

    Args:
        hand_actions (dict): {hand_group: most_common_action} 형태의 데이터.
                             most_common_action은 해당 핸드에서 가장 많이 선택된 액션.
    """
    # 카드 랭크 정의
    ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
    num_ranks = len(ranks)
    grid = np.zeros((num_ranks, num_ranks))  # 그리드 초기화

    # 핸드 데이터 채우기
    for hand_group, most_common_action in hand_actions.items():
        if len(hand_group) == 2:  # 페어
            rank1, rank2 = hand_group[0], hand_group[1]
            row = ranks.index(rank1)
            col = ranks.index(rank2)
            if row != col:
                continue  # 페어 외에는 무시
        elif len(hand_group) == 3:  # 수딧/오프수딧 핸드
            rank1, rank2, suit = hand_group[0], hand_group[1], hand_group[2]
            
            if rank1 < rank2:
                if suit == 's':
                    row = ranks.index(rank1)
                    col = ranks.index(rank2)
                else:
                    col = ranks.index(rank1)
                    row = ranks.index(rank2)
            else:
                if suit == 's':
                    col = ranks.index(rank1)
                    row = ranks.index(rank2)
                else:
                    row = ranks.index(rank1)
                    col = ranks.index(rank2)

            if suit == 's':  # 수딧 핸드
                if row >= col:
                    continue  # 수딧 핸드는 오른쪽 위에만 표시
            elif suit == 'o':  # 오프수딧 핸드
                if row <= col:
                    continue  # 오프수딧 핸드는 왼쪽 아래에만 표시

        # 그리드에 액션 기록
        grid[row, col] = most_common_action

    # 플롯
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(grid, cmap='coolwarm', origin='upper')

    # 핸드 텍스트 표시
    for i in range(num_ranks):
        for j in range(num_ranks):
            if i <= j:  # 오른쪽 위(수딧 핸드와 페어)
                hand_label = f"{ranks[i]}{ranks[j]}s" if i != j else f"{ranks[i]}{ranks[j]}"
            else:  # 왼쪽 아래(오프수딧 핸드)
                hand_label = f"{ranks[j]}{ranks[i]}o"
            ax.text(j, i, hand_label, ha='center', va='center', color='black')

    # 설정
    ax.set_xticks(range(num_ranks))
    ax.set_yticks(range(num_ranks))
    ax.set_xticklabels(ranks)
    ax.set_yticklabels(ranks)
    ax.set_xlabel('Opponent Card Rank')
    ax.set_ylabel('Your Card Rank')
    plt.title('Hand Range Visualization')

    # 컬러바 추가
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Most Frequent Action')

    plt.show()

class Stage(Enum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    END_HIDDEN = 4
    SHOWDOWN = 5




def traincs(args, public_cards = None):


    # Check whether gpu is available
    device = get_device()
        
    # Seed numpy, torch, random
    set_seed(int(time.time()))

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
        from rlcard.agents import DQNAgent
        agent2 = DQNAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            mlp_layers=[64,64],
            device=device,
        )

    load_path = os.path.join(args.log_dircs, 'model.pth')
    agent = torch.load(load_path)

    load_path = os.path.join(args.log_dir, 'model.pth')
    agent2 = torch.load(load_path)

    agents = [agent, agent2]
    env.set_agents(agents)
    #state = {'hand': ['H5', 'H6'], 'public_cards': ['D3', 'HQ', 'C3'], 'all_chips': [4, 4], 'my_chips': 4, 'legal_actions': [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT, Action.RAISE_POT, Action.ALL_IN], 'stakes': [96, 96], 'current_player': 0, 'pot': 8, 'stage': Stage.FLOP}
    action_record = [(0, Action.RAISE_POT), (1, Action.CHECK_CALL), (1, Action.CHECK_CALL)]

    hand_actions = evaluate_hands(agent, public_cards, action_record, env)
    plot_hand_range(hand_actions)

    hand_actions = evaluate_hands(agent2, public_cards, action_record, env)

    # 핸드레인지 시각화
    plot_hand_range(hand_actions)

def trainma(args, public_cards = None):


    # Check whether gpu is available
    device = get_device()
        
    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(
        args.env,
        config={
            'seed': 43,
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
        from rlcard.agents import DQNAgent
        agent2 = DQNAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            mlp_layers=[64,64],
            device=device,
        )

    load_path = os.path.join(args.log_dirma, 'model.pth')
    agent = torch.load(load_path)

    load_path = os.path.join(args.log_dir, 'model.pth')
    agent2 = torch.load(load_path)

    agents = [agent, agent2]
    env.set_agents(agents)
    #state = {'hand': ['H5', 'H6'], 'public_cards': ['D3', 'HQ', 'C3'], 'all_chips': [4, 4], 'my_chips': 4, 'legal_actions': [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT, Action.RAISE_POT, Action.ALL_IN], 'stakes': [96, 96], 'current_player': 0, 'pot': 8, 'stage': Stage.FLOP}
    action_record = [(0, Action.RAISE_POT), (1, Action.CHECK_CALL), (1, Action.CHECK_CALL)]

    hand_actions = evaluate_hands(agent, public_cards, action_record, env)

    # 핸드레인지 시각화
    plot_hand_range(hand_actions)

    hand_actions = evaluate_hands(agent2, public_cards, action_record, env)

    # 핸드레인지 시각화
    plot_hand_range(hand_actions)

    




'''
    state, player_id = env.set_postflop(Card('H','5'), Card('H','6'))
    print("start first game!!!!")
    print("dealer is : %d" % env.game.dealer_id)
    
    if player_id == 0:
        next_state, next_player_id = env.step(3, env.agents[player_id].use_raw)
        state = next_state
        player_id = next_player_id
    else:
        next_state, next_player_id = env.step(3)
        state = next_state
        player_id = next_player_id
    if player_id == 0:
        next_state, next_player_id = env.step(1, env.agents[player_id].use_raw)
        state = next_state
        player_id = next_player_id
    else:
        next_state, next_player_id = env.step(1)
        state = next_state
        player_id = next_player_id

    

    while not env.is_over():
        print(player_id)
        if player_id == 0:
            print(state)
            action, info = env.agents[player_id].eval_step(state)
            print(info)
            next_state, next_player_id = env.step(action, env.agents[player_id].use_raw)
            state = next_state
            player_id = next_player_id
        else:
            action = env.agents[player_id].step(state)
            next_state, next_player_id = env.step(action)
            state = next_state
            player_id = next_player_id
'''


'''
    if state['stage'] == "<stage.PREFLOP: 0>":
        if player_id == 0:
            action, _ = env.agents[player_id].eval_step(state)
            next_state, next_player_id = env.step(action, env.agents[player_id].use_raw)
            state = next_state
            player_id = next_player_id
        else:
            action = env.agents[player_id].step(state)
            next_state, next_player_id = env.step(action, env.agents[player_id].use_raw)
            state = next_state
            player_id = next_player_id
'''


'''
    while not env.is_over():
        print(player_id)
        if player_id == 0:
            action, _ = env.agents[player_id].eval_step(state)
            next_state, next_player_id = env.step(action, env.agents[player_id].use_raw)
            state = next_state
            player_id = next_player_id
            print(state)
        else:
            action = env.agents[player_id].step(state)
            next_state, next_player_id = env.step(action, env.agents[player_id].use_raw)
            state = next_state
            player_id = next_player_id
            print(state)

'''
'''
    'raw_obs': {'hand': ['S3', 'D3'], 
                'public_cards': [], 
                'all_chips': [1, 2], 
                'my_chips': 1, 
                'legal_actions': [<Action.FOLD: 0>, <Action.CHECK_CALL: 1>, <Action.RAISE_POT: 3>, <Action.ALL_IN: 4>], 
                'stakes': [99, 98], 
                'current_player': 0, 
                'pot': 3, 
                'stage': <Stage.PREFLOP: 0>}, 
                'raw_legal_actions': [<Action.FOLD: 0>, <Action.CHECK_CALL: 1>, <Action.RAISE_POT: 3>, <Action.ALL_IN: 4>], 
                'action_record': []}


    {'raw_obs': {'hand': ['SK', 'HQ'], 
                'public_cards': ['C3', 'SJ', 'H9'], 
                'all_chips': [4, 4], 'my_chips': 4, 
                'legal_actions': [<Action.FOLD: 0>, <Action.CHECK_CALL: 1>, <Action.RAISE_HALF_POT: 2>, <Action.RAISE_POT: 3>, <Action.ALL_IN: 4>], 
                'stakes': [96, 96], 
                'current_player': 1, 
                'pot': 8, 'stage': <Stage.FLOP: 1>}, 
    'raw_legal_actions': [<Action.FOLD: 0>, <Action.CHECK_CALL: 1>, <Action.RAISE_HALF_POT: 2>, <Action.RAISE_POT: 3>, <Action.ALL_IN: 4>], 
    'action_record': [(0, <Action.RAISE_POT: 3>), (1, <Action.CHECK_CALL: 1>)]}

'''




    

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
    train(args)


























'''
community_cards = ['Ac', 'Kd', '10h']
pot = 50
stage = 'FLOP'
hand_actions = evaluate_hands(agent, community_cards, pot, stage)

# 핸드레인지 시각화
plot_hand_range(hand_actions)
'''