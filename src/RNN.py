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
    RNN is a single-layer (stack) RNN by connecting multiple RNNCell together in a single
    direction, where the input sequence is processed from left to right.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Constructor of the RNN module.

        Inputs:
        - input_dim: Dimension of the input x_t
        - hidden_dim: Dimension of the hidden state h_{t-1} and h_t
        """
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim

        
        self.rnn_cell = RNNCell(input_dim, hidden_dim)
        

    def forward(self, x: torch.Tensor):
        """
        Compute the hidden representations for every token in the input sequence.

        Input:
        - x: A tensor with the shape of BxLxC, where B is the batch size, L is the squence
          length, and C is the channel dimmension

        Return:
        - h: A tensor with the shape of BxLxH, where H is the hidden dimension of RNNCell
        """
        b = x.shape[0]
        seq_len = x.shape[1]

        # initialize the hidden dimension
        init_h = x.new_zeros((b, self.hidden_dim))

        h = None
       
        h = torch.zeros(b, seq_len, self.hidden_dim, device=x.device)
        h_prev = init_h
        for i in range(seq_len):
            h_prev = self.rnn_cell(x[:, i, :], h_prev)
            h[:, i, :] = h_prev
        

        return h

class RNNQNetwork(nn.Module):
    """
    RNNQNetwork is a Q-network that uses an RNN to process the input sequence and a linear layer to output the Q-values.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Constructor of the RNNQNetwork.

        Inputs:
        - input_dim: Dimension of the input x_t
        - hidden_dim: Dimension of the hidden state h_{t-1} and h_t
        - output_dim: Dimension of the output Q-values
        """
        super(RNNQNetwork, self).__init__()
        
        self.rnn = RNN(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        """
        Compute the Q-values for the input sequence.
        """
        h = self.rnn(x)
        q = self.linear(h)
        return q

if __name__ == "__main__":
    x = torch.randn(1, 10, 3)
    rnn_q_network = RNNQNetwork(input_dim=3, hidden_dim=10, output_dim=3)
    q = rnn_q_network(x)


    print(q.shape)