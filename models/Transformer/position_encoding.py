import numpy as np
import torch
from torch import nn


class PositionEmbeddingStandard(nn.Module):
    """
    Standard position embedding used in the 'Attention is all you need' Transformer
    """

    def __init__(self, d_hid, n_position=200):
        """

        :param d_hid: token dimension of the input tensor
        :param n_position: length of the sequence
        """
        super(PositionEmbeddingStandard, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """

        :param d_hid: token dimension of the input tensor
        :param n_position: length of the sequence
        :return: position encoding
        """

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        """

        :param x: tensor for which a position encoding is needed
        :return: position encoding for tensor x
        """
        return self.pos_table[:, :x.size(1)].clone().detach()
