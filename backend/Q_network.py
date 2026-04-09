import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    """
    Dueling Q-Network with noisy layers, layer normalization, and orthogonal init.

    Architecture improvements over baseline:
      - Dueling streams: separates state value V(s) from advantage A(s,a),
        which stabilizes learning by decoupling "how good is this state"
        from "how good is each action".
      - NoisyLinear output heads: parametric noise replaces epsilon-greedy
        exploration, enabling state-dependent adaptive exploration.
      - LayerNorm after each hidden layer: reduces internal covariate shift
        and stabilizes training, especially helpful for non-stationary targets.
      - Orthogonal initialization: better gradient flow than naive normal init.
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

        self._init_weights()

    def _init_weights(self):
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state : torch.Tensor  shape (..., state_size)

        Returns
        -------
        torch.Tensor  shape (..., action_size)
            Q-values for each action.
        """

        features = self.trunk(state)

        value = self.value_stream(features)                        # (..., 1)
        advantage = self.advantage_stream(features)                # (..., action_size)

        # Combine: Q(s,a) = V(s) + A(s,a) - mean_a[A(s,a)]
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q

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
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

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