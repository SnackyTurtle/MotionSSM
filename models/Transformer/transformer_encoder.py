from .position_encoding import PositionEmbeddingStandard
from .mlp import MLP

import torch
import torch.nn as nn
import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=512, dropout=0.1, n_heads=8) -> None:
        super().__init__()
        # self attention
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, query_pos, padding_mask, self_attn_mask):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        assert torch.isfinite(q).all()
        assert torch.isfinite(k).all()
        assert torch.isfinite(tgt).all()
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(
            0, 1), key_padding_mask=padding_mask, attn_mask=self_attn_mask)[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # feed forward
        tgt = self.forward_ffn(tgt)
        return tgt

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos


class MotionTransformer(nn.Module):
    """Motion Prediction Module from 'MotionTrack: Learning Motion Predictor for Multiple Object Tracking'. This is in no way
    related to the more prominemt 'MotionTrack: Learning Robust Short-term and Long-term Motions for Multi-Object Tracking'
    """

    def __init__(self, num_layers = 6, seq_len = 10) -> None:
        super().__init__()
        encoder_layer = TransformerEncoderLayer()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.seq_len = seq_len

        self.num_feats = 8


        self.input_embed = nn.Linear(self.num_feats, 256)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.prediction_head = MLP(256, 256, 4, 3)
        self.pos_enc = PositionEmbeddingStandard(256, seq_len)

    def forward(self, seq_features):
        seq_features = seq_features.to(torch.float32)
        if seq_features.dim() == 2:
            seq_features = seq_features.unsqueeze(1)
        bs, _, _ = seq_features.shape
        output = self.input_embed(seq_features)

        query_pos = self.pos_enc(output)


        for lid, layer in enumerate(self.layers):
            output = layer(output, query_pos, padding_mask=None, self_attn_mask=None)

        output = torch.mean(output, dim=1)
        offset = self.prediction_head(output)

        pred_box = offset + seq_features[:,-1,:4]

        return pred_box
