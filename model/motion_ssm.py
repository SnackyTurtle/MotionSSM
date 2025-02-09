import torch
from model.flow_ssm_decoder import SSMDecoder
from  model.mamba_encoder import MambaEncoder

class MotionSSM(torch.nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.config = config
        self.device = device

        self.encoder = MambaEncoder(d_model=256,n_layer=6,vocab_size=8,rms_norm=True,fused_add_norm=False)
        self.decoder = SSMDecoder()

    def forward(self, x):
        out_enc = self.encoder(x['seq_features'])
        bbox_pred, bbox_interm = self.decoder(x['seq_features'][:, -1, :4], out_enc)
        return bbox_pred, bbox_interm
