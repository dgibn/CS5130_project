import torch
import torch.nn as nn


class RNNCell(torch.nn.Module):
    """
    RNNCell is a single cell that takes x_t and h_{t_1} as input and outputs h_t.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Constructor of RNNCell.

        Inputs:
        - input_dim: Dimension of the input x_t
        - hidden_dim: Dimension of the hidden state h_{t-1} and h_t
        """

        # We always need to do this step to properly implement the constructor
        super(RNNCell, self).__init__()

        self.linear_x, self.linear_h, self.non_linear = None, None, None

        
        self.linear_x = nn.Linear(input_dim, hidden_dim)
        self.linear_h = nn.Linear(hidden_dim, hidden_dim)
        self.non_linear = nn.Tanh()
       

    def forward(self, x_cur: torch.Tensor, h_prev: torch.Tensor):
        """
        Compute h_t given x_t and h_{t-1}.

        Inputs:
        - x_cur: x_t, a tensor with the same of BxC, where B is the batch size and
          C is the channel dimension.
        - h_prev: h_{t-1}, a tensor with the same of BxH, where H is the channel
          dimension.
        """
        h_cur = None
       
        x_linear = self.linear_x(x_cur)
        h_linear = self.linear_h(h_prev)
        h_cur = self.non_linear(x_linear + h_linear)
        
        return h_cur

class RNN(torch.nn.Module):
    """
    Multi-layer RNN by stacking RNNCells. The output of layer l at timestep t
    becomes the input to layer l+1 at the same timestep.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1):
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.cells.append(RNNCell(in_dim, hidden_dim))

    def forward(self, x: torch.Tensor):
        """
        Input:  x with shape BxLxC
        Return: h with shape BxLxH (hidden states from the last layer)
        """
        b, seq_len, _ = x.shape
        h_prev = [x.new_zeros(b, self.hidden_dim) for _ in range(self.num_layers)]
        h_out = torch.zeros(b, seq_len, self.hidden_dim, device=x.device)

        for t in range(seq_len):
            inp = x[:, t, :]
            for l, cell in enumerate(self.cells):
                h_prev[l] = cell(inp, h_prev[l])
                inp = h_prev[l]
            h_out[:, t, :] = h_prev[-1]

        return h_out

class RNNQNetwork(nn.Module):
    """
    RNNQNetwork is a Q-network that uses an RNN to process the input sequence and a linear layer to output the Q-values.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 1):
        super(RNNQNetwork, self).__init__()

        self.rnn = RNN(input_dim, hidden_dim, num_layers)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        h = self.rnn(x)
        q = self.linear(h)
        return q


if __name__ == "__main__":
    x = torch.randn(1, 10, 8)
    rnn_q_network = RNNQNetwork(input_dim=8, hidden_dim=128, output_dim=3, num_layers=4)
    q = rnn_q_network(x)
    print(q.shape)