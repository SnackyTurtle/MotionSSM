import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    yery simple multi-layer perceptron (also called FFN)
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        """

        :param input_dim: dimension of the expected input
        :param hidden_dim: dimension of the hidden layers
        :param output_dim: dimension of the wanted output
        :param num_layers: number of hidden layers - 1
        """
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
