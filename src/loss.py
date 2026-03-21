import torch
import torch.nn as nn
import torch.nn.functional as F

def double_dqn_loss(q_values, actions, rewards,
                    next_q_online, next_q_target, dones, gamma=0.95):
    """
    Online network selects the best next action;
    target network evaluates it -- breaks the maximization bias.
    """
    q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        best_next_actions = next_q_online.argmax(dim=1, keepdim=True)
        next_q = next_q_target.gather(1, best_next_actions).squeeze(1)
        target = rewards + gamma * next_q * (1.0 - dones.float())

    return F.smooth_l1_loss(q_selected, target)
