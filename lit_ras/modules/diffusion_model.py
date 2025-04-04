import torch
import numpy as np

# Typing
from torch import Tensor
from typing import Tuple, List, Optional

####################### Base modules #######################

from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

from graphite.nn.basis import GaussianRandomFourierFeatures

class PositionalEncoding(nn.Module):
    """Positional encoding layer from pytorch.org/tutorials/beginner/transformer_tutorial.html.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, scale:float = 1.0):
        """
        Args:
            d_model (int): Embedding dimension.
            dropout (float): Dropout probability.
            max_len (int): Sequence length.
            scale (float): Scaling factor such that the positional embedding values
                are relatively small compared to the original embedding values.
        """
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Inputs with shape (batch_size, seq_len, embed_dim)
        """
        x = x + self.scale * self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerBackbone(nn.Module):
    def __init__(self, init_dim: int, dim: int, ff_dim: int, num_heads: int, num_layers: int, ucg_sizes: List[int], dropout: float):
        super().__init__()
        self.init_dim   = init_dim
        self.dim        = dim
        self.ff_dim     = ff_dim
        self.num_heads  = num_heads
        self.num_layers = num_layers
        self.ucg_sizes  = ucg_sizes
        self.dropout    = dropout
        
        self.embed_token = nn.Linear(init_dim, dim)
        self.embed_time = GaussianRandomFourierFeatures(dim, input_dim=1)
        self.pos_encoder = PositionalEncoding(dim, dropout)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer = TransformerEncoderLayer(dim, num_heads, ff_dim, dropout, batch_first=True),
            num_layers = num_layers,
        )
        self.lin_out = nn.Linear(dim, init_dim-3)

    def _process_input(self, ucg_pos: Tensor, cg_disp: Tensor) -> Tensor:
        ucg_sizes = self.ucg_sizes[:ucg_pos.shape[1]]
        split_cg_disp = cg_disp.split(ucg_sizes, dim=1)
        
        pad_ammts = max(ucg_sizes) - np.array(ucg_sizes)
        padded = [
            F.pad(split_cg_disp[i], pad=(0, 0, 0, pad_ammt), mode='constant', value=0)
            for i, pad_ammt in enumerate(pad_ammts)
        ]

        combined = torch.cat([ucg_pos.unsqueeze(2), torch.stack(padded, dim=1)], dim=2)
        B, N, _, _ = combined.shape
        return combined.view(B, N, -1)
    
    def _final_output(self, transformer_out: Tensor) -> Tensor:
        B, N, _ = transformer_out.shape
        ucg_sizes = self.ucg_sizes[:N]

        final = []
        for i, ucg_size in enumerate(ucg_sizes):
            final.append(
                transformer_out[:, i, :].view(B, -1, 3)[:, :ucg_size, :]
            )
        return torch.cat(final, dim=1)

    def forward(self, ucg_pos: Tensor, cg_disp: Tensor, t: Tensor) -> Tensor:
        inputs = self._process_input(ucg_pos, cg_disp)
        inputs = self.embed_token(inputs)
        inputs = self.pos_encoder(inputs)
        
        src = torch.cat([self.embed_time(t), inputs], dim=1)
        out = self.transformer_encoder(src)
        out = self.lin_out(out)[:, 1:, :]
        return self._final_output(out)

####################### LightningModule #######################

import pytorch_lightning as L
import torch.nn.functional as F

from graphite.diffusion import VariancePreservingDiffuser

class LitUCG2CGNoiseNet(L.LightningModule):
    def __init__(self, init_dim: int, dim: int, ff_dim: int, num_heads: int, num_layers: int, ucg_index_file: int, dropout: float, learn_rate: float):
        super().__init__()
        self.save_hyperparameters()

        # Read ucg sizes
        ucg_sizes = []
        data = np.load(ucg_index_file, allow_pickle=True)
        data = data["indices_per_cluster"]
        for i in range(data.shape[0]):
            ucg_sizes.append(len(data[i]))
        
        # with open(ucg_index_file, 'r') as f:
        #     for line in f:
        #         ucg_sizes.append(len(line.rstrip().split(',')))


        # Core model
        self.model = TransformerBackbone(
            init_dim   = init_dim,
            dim        = dim,
            ff_dim     = ff_dim,
            num_heads  = num_heads,
            num_layers = num_layers,
            ucg_sizes  = ucg_sizes,
            dropout    = dropout,
        )

        # Data diffuser
        self.diffuser = VariancePreservingDiffuser()

        # Parameters
        self.learn_rate = learn_rate

    def forward(self, ucg_pos: Tensor, cg_disp: Tensor, t: Tensor) -> Tensor:
        return self.model(ucg_pos, cg_disp, t)

    def _get_loss(self, ucg_pos, cg_disp):
        t = torch.empty(ucg_pos.size(0), 1, 1, device=cg_disp.device).uniform_(self.diffuser.t_min, self.diffuser.t_max)
        noisy_cg_disp, eps = self.diffuser.forward_noise(cg_disp, t)
        pred_eps = self.model(ucg_pos, noisy_cg_disp, t)
        return F.mse_loss(pred_eps, eps)

    def _get_combined_loss(self, combined_batch):
        losses = []
        for sub_batch in combined_batch.values():
            if sub_batch is not None:
                losses.append(self._get_loss(sub_batch['ucg_pos'], sub_batch['cg_disp']))
        return 1 / len(losses) * sum(l for l in losses)

    def training_step(self, batch, *args):
        loss = self._get_loss(batch['ucg_pos'], batch['cg_disp'])
        bsize = batch['ucg_pos'].size(0)
        self.log('train_loss', loss, prog_bar=True, batch_size=bsize, logger=True, on_epoch=True, on_step=True, sync_dist=True) # TODO: (sync_dist=) but causes extra overhead
        return loss

    def validation_step(self, batch, *args):
        loss = self._get_loss(batch['ucg_pos'], batch['cg_disp'])
        bsize = batch['ucg_pos'].size(0)
        self.log('valid_loss', loss, prog_bar=True, batch_size=bsize, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learn_rate)
    
    def generate(self, ucg_pos, num_steps):
        device  = ucg_pos.device
        B       = ucg_pos.shape[0]
        # num_cgs = 2897
        num_cgs = 751
        # if ucg_pos.size(1) == 100: num_cgs = 1819
        # if ucg_pos.size(1) == 250: num_cgs = 4716
        
        ts     = torch.linspace(self.diffuser.t_max, self.diffuser.t_min, num_steps+1, device=device).view(-1, 1, 1, 1)
        ts     = ts.expand(-1, B, 1, 1)
        x      = torch.randn(B, num_cgs, 3, device=device)
        x_traj = [x.clone()]
        
        f, g, g2 = self.diffuser.f, self.diffuser.g, self.diffuser.g2
        sigma = self.diffuser.sigma
        for i, t in enumerate(ts[1:]):
            dt = ts[i+1] - ts[i]
            eps = dt.abs().sqrt() * torch.randn_like(x)
            pred_eps = self.model(ucg_pos, x, t)
            dx = (f(t)*x + g2(t)*pred_eps/sigma(t))*dt + g(t)*eps
            x += dx
            x_traj.append(x.clone())

        return torch.stack(x_traj, dim=1)
