import os
import random
import argparse
import numpy as np
import rlcard
from rlcard.agents.dqn_agent import DQNAgent
from rlcard.agents.cfr_agent import CFRAgent
from rlcard.agents.random_agent import RandomAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve
)


# 랜덤 시드 설정
set_seed(42)

# No-Limit Hold'em 환경 생성
env = rlcard.make('no-limit-holdem', config={'allow_step_back' : True, 'seed' : 42})

print(env.num_actions)

device = get_device()

cfr_agent = CFRAgent(env)
'''
print("Training CFR Agent...")
for _ in range(100):  # 100번 반복 학습
    cfr_agent.train()

print("CFR Agent Training Complete!")

env.set_agents([None, cfr_agent])





print("Starting game against CFR Agent...")
state, player_id = env.reset()

while not env.is_over():
    if player_id == 0:  # 사용자의 턴
        print("\nYour turn:")
        print(f"Your state: {state}")
        print(f"Legal actions: {state['legal_actions']}")
        
        # 사용자로부터 행동 입력받기
        try:
            user_action = int(input("Choose your action: "))
            if user_action not in state['legal_actions']:
                raise ValueError
        except ValueError:
            print("Invalid action. Please choose from the legal actions.")
            continue
    else:  # CFR 에이전트의 턴
        user_action = cfr_agent.step(state)
        print(f"CFR Agent chose action: {user_action}")

    # 행동 수행
    state, player_id = env.step(user_action)

# 결과 출력
print("\nGame Over!")
payoffs = env.get_payoffs()
if payoffs[0] > payoffs[1]:
    print(f"You win! Your payoff: {payoffs[0]}, CFR Agent payoff: {payoffs[1]}")
elif payoffs[0] < payoffs[1]:
    print(f"You lose! Your payoff: {payoffs[0]}, CFR Agent payoff: {payoffs[1]}")
else:
    print(f"It's a draw! Your payoff: {payoffs[0]}, CFR Agent payoff: {payoffs[1]}")
'''