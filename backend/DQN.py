import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQN(nn.Module):
    """
    Dueling Q-Network with:
      - Noisy layers + layer normalization + orthogonal init  (unchanged)
      - Dueling streams: V(s) and A(s,a)                     (unchanged)
      - Auxiliary volume head: classifies next-step trade volume into    (NEW)
        one of 4 discrete share buckets: {5, 10, 15, 20}

    The shared trunk is trained jointly by two loss signals:
      1. TD loss from the RL objective        (Q-values)
      2. CrossEntropyLoss from volume head    (share-bucket classification)

    This auxiliary task forces the trunk to encode volume dynamics,
    improving both exploration quality and sample efficiency.

    Inputs
    ------
    state_size  : total number of input features per timestep
                  e.g. OHLCV (5) + any technical indicators
    action_size : number of discrete actions the agent can take
    hidden_size : width of the shared trunk layers

    Forward outputs
    ---------------
    q_values      : Tensor (..., action_size)  — Q-value per action
    volume_logits : Tensor (..., 4)            — raw logits over [5,10,15,20] share buckets

    Training tip
    ------------
    Volume targets must be class indices (LongTensor), not raw share counts:
        vol_targets = torch.tensor([vol_class_map[v] for v in raw_shares])
    Combined loss example:
        loss = td_loss + lambda_vol * F.cross_entropy(volume_logits, vol_targets)
    A lambda_vol of 0.1–0.5 works well as a starting point.
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.action_size = action_size

        # --- Shared feature trunk ---
        self.trunk = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        # --- Dueling heads (noisy for exploration) ---
        # Value stream: V(s) — scalar
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            NoisyLinear(hidden_size // 2, 1),
        )

        # Advantage stream: A(s, a) — one per action
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            NoisyLinear(hidden_size // 2, action_size),
        )

        # --- Auxiliary volume classification head (NEW) ---
        # Classifies next-step trade volume into one of NUM_VOL_CLASSES
        # discrete buckets: {5, 10, 15, 20} shares.
        # Outputs raw logits — apply softmax externally or use CrossEntropyLoss
        # directly (which expects logits, not probabilities).
        self.volume_classes = [5, 10, 15, 20]          # ordered share buckets
        num_vol_classes     = len(self.volume_classes)  # 4

        self.volume_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_vol_classes),  # 4 logits
        )

        self._init_weights()

    def _init_weights(self):
        # Orthogonal init for trunk
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(m.bias)

        # Orthogonal init for volume head too (NEW)
        for m in self.volume_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        state : torch.Tensor  shape (..., state_size)
            Mixed feature vector: OHLCV + technical indicators, normalized.

        Returns
        -------
        q_values     : torch.Tensor  shape (..., action_size)
        volume_logits: torch.Tensor  shape (..., 4)
            Raw logits over share buckets [5, 10, 15, 20].
            To get predicted bucket : volume_logits.argmax(-1)
            To get predicted shares : model.volume_classes[volume_logits.argmax(-1)]
        """
        features = self.trunk(state)

        # RL heads
        value     = self.value_stream(features)                        # (..., 1)
        advantage = self.advantage_stream(features)                    # (..., action_size)
        q_values  = value + advantage - advantage.mean(dim=-1, keepdim=True)

        # Auxiliary volume classification (NEW)
        volume_logits = self.volume_head(features)                     # (..., 4)

        return q_values, volume_logits

    def reset_noise(self):
        """Call once per training step to resample noise in NoisyLinear layers."""
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


class NoisyLinear(nn.Module):
    """
    Linear layer with factorised Gaussian noise for parameter-space exploration.
    Replaces epsilon-greedy with learned, state-dependent exploration.
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.sigma_init   = sigma_init

        # Learnable parameters
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))

        # Noise buffers (not parameters — resampled each step)
        self.register_buffer("weight_eps", torch.empty(out_features, in_features))
        self.register_buffer("bias_eps",   torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = self.in_features ** -0.5
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma_init * bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(self.sigma_init * bound)

    def reset_noise(self):
        eps_in  = self._f(torch.randn(self.in_features,  device=self.weight_mu.device))
        eps_out = self._f(torch.randn(self.out_features, device=self.weight_mu.device))
        self.weight_eps.copy_(eps_out.outer(eps_in))
        self.bias_eps.copy_(eps_out)

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        return x.sign() * x.abs().sqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_eps
            b = self.bias_mu   + self.bias_sigma   * self.bias_eps
        else:
            w, b = self.weight_mu, self.bias_mu
        return F.linear(x, w, b)


# ---------------------------------------------------------------------------
# Example training loop sketch
# ---------------------------------------------------------------------------
#
# model     = DuelingDQN(state_size=20, action_size=3)  # e.g. buy/hold/sell
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# LAMBDA    = 0.2   # weight for volume auxiliary loss
#
# vol_class_map = {5: 0, 10: 1, 15: 2, 20: 3}   # shares -> class index
#
# for states, actions, rewards, next_states, dones, vol_targets in dataloader:
#     # vol_targets: LongTensor of class indices (0–3), shape (batch,)
#     # e.g. 10 shares -> index 1
#
#     q_values, vol_logits = model(states)
#
#     # --- TD loss (your existing logic) ---
#     with torch.no_grad():
#         q_next, _ = model(next_states)
#     td_target = rewards + 0.99 * q_next.max(-1).values * (1 - dones)
#     td_loss   = F.smooth_l1_loss(q_values.gather(1, actions.unsqueeze(1)), td_target.unsqueeze(1))
#
#     # --- Volume classification loss (NEW) ---
#     vol_loss = F.cross_entropy(vol_logits, vol_targets)   # expects raw logits
#
#     loss = td_loss + LAMBDA * vol_loss
#
#     optimizer.zero_grad()
#     loss.backward()
#     torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
#     optimizer.step()
#     model.reset_noise()
#
# # --- Inference: decode predicted share count ---
# _, vol_logits  = model(state)
# pred_idx       = vol_logits.argmax(-1)                     # 0–3
# pred_shares    = model.volume_classes[pred_idx.item()]     # 5/10/15/20