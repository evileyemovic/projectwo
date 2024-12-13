import os
import argparse
import time
import torch

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

print("GO!")

def train(args):

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
        agent1 = DQNAgent_CS(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            mlp_layers=[64,64],
            device=device,
        )

    agents = [agent1, agent1]
    env.set_agents(agents)

    # Start training
    logger1 = Logger(args.log_dir)  # agent0 로그 디렉토리

    with logger1:  # 두 Logger를 함께 관리
        for episode in range(args.num_episodes):

            if args.algorithm == 'nfsp':
                agents[0].sample_episode_policy()

            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)

            # Feed transitions into agent memory, and train the agent
            # Here, we assume that DQN always plays the first position
            # and the other players play randomly (if any)

            for ts in trajectories[0]:
                if ts[1] == 1:  # Check, Call시 보상 증가
                    ts[2] += 2
                elif ts[1] == 0:
                    ts[2] -= 1  # Fold에 페널티
                elif ts[1] in [2, 3, 4]:
                    ts[2] -= 0.5
                agent1.feed(ts)
            for ts in trajectories[1]:
                if ts[1] == 1:  # Check, Call시 보상 증가
                    ts[2] += 2
                elif ts[1] == 0:
                    ts[2] -= 1  # Fold에 페널티
                elif ts[1] in [2, 3, 4]:
                    ts[2] -= 0.5
                agent1.feed(ts)


        # 성능 평가
            if episode % args.evaluate_every == 0:
            # agent0 성능 평가 및 로그 기록
                agent1_performance = tournament(env, args.num_eval_games)[0]
                logger1.log_performance(episode, agent1_performance)

    # 최종 CSV 및 그래프 파일 경로 출력
        csv_path1, fig_path1 = logger1.csv_path, logger1.fig_path

    # Plot the learning curve
    plot_curve(csv_path1, fig_path1, args.algorithm)

    # Save model
    save_path = os.path.join(args.log_dir, 'model.pth')
    torch.save(agent1, save_path)


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
        default=50000,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=2000,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=5000,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/Base_CS2/',
    )
    parser.add_argument(
        '--log_dir1',
        type=str,
        default='experiments/Advance_CS2_1/',
    )
    parser.add_argument(
        '--log_dir2',
        type=str,
        default='experiments/Advance_CS2_2/',
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)