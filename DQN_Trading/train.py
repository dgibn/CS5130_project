"""
Training loop for the DQN trading agent.
"""

import random
import numpy as np

from agent import DQNAgent
from env import TradingEnv


def train(
    agent: DQNAgent,
    env: TradingEnv,
    n_episodes: int = 500,
    max_steps: int | None = None,
    verbose: bool = True,
    save_every: int = 100,
):
    """
    Train the DQN agent.

    Parameters
    ----------
    agent       : DQNAgent instance
    env         : TradingEnv instance (training data)
    n_episodes  : number of training episodes
    max_steps   : max days per episode (None = full dataset)
    verbose     : print progress every 50 episodes
    save_every  : save a checkpoint every N episodes

    Returns
    -------
    dict with keys: rewards, losses, portfolio_values
    """
    if max_steps is None:
        max_steps = env.n_steps - 1

    reward_history = []
    loss_history = []
    pv_history = []

    for ep in range(1, n_episodes + 1):
        # start at a random point for data diversity
        start = random.randint(0, max(0, env.n_steps - max_steps - 1))
        state = env.reset(start_idx=start)
        total_reward = 0.0
        ep_losses = []

        for _ in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.memory.push(state, action, reward, next_state, done)

            loss = agent.learn()
            if loss is not None:
                ep_losses.append(loss)

            agent.update_target()

            state = next_state
            total_reward += reward

            if done:
                break

        agent.decay_epsilon()

        avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        reward_history.append(total_reward)
        loss_history.append(avg_loss)
        pv_history.append(info["portfolio_value"])

        if verbose and ep % 50 == 0:
            print(
                f"Episode {ep:4d} | Reward: {total_reward:+.4f} | "
                f"Loss: {avg_loss:.6f} | PV: ${info['portfolio_value']:,.2f} | "
                f"ε: {agent.epsilon:.4f}"
            )

        if save_every and ep % save_every == 0:
            agent.save(f"checkpoints/dqn_ep{ep}.pt")

    # final save
    agent.save("checkpoints/dqn_final.pt")

    return {
        "rewards": reward_history,
        "losses": loss_history,
        "portfolio_values": pv_history,
    }