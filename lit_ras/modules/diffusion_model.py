
"""
diffusion_model.py

Defines the Transformer-based diffusion model for UCG-to-CG backmapping:
- PositionalEncoding for sequence inputs
- TransformerBackbone for noise prediction
- LitUCG2CGNoiseNet (PyTorch Lightning Module) for training, validation, and generation
"""

import torch
import numpy as np
from torch import nn, Tensor
from typing import Tuple, List, Optional

import math
import torch.nn.functional as F
import pytorch_lightning as L

from graphite.nn.basis import GaussianRandomFourierFeatures
from graphite.diffusion import VariancePreservingDiffuser
from torch.nn import TransformerEncoder, TransformerEncoderLayer

####################### Positional Encoding Module #######################

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding module.

    Adds position-dependent signals to input embeddings to enable Transformers
    to capture sequential order information.
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
        Add positional encoding to input.

        Args:
            x (Tensor): Input tensor (batch_size, seq_len, embed_dim).

        Returns:
            Tensor: Positionally encoded input.
        """
        x = x + self.scale * self.pe[:, :x.size(1), :]
        return self.dropout(x)

####################### Transformer Backbone #######################

class TransformerBackbone(nn.Module):
    """
    Transformer-based neural network backbone for noise prediction
    in UCG-to-CG diffusion models.
    """

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
        self.lin_out = nn.Linear(dim, init_dim - 3)

    def _process_input(self, ucg_pos: Tensor, cg_disp: Tensor) -> Tensor:
        """
        Prepares input by splitting and padding CG displacements.

        Args:
            ucg_pos (Tensor): UCG positions (B, 40, 3).
            cg_disp (Tensor): CG displacements (B, 751, 3).

        Returns:
            Tensor: Prepared input (B, num_UCG_beads, features).
        """
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
        """
        Reshape final Transformer output back into CG bead displacements.

        Args:
            transformer_out (Tensor): Transformer output (B, seq_len, embed_dim).

        Returns:
            Tensor: Predicted CG displacements (B, 751, 3).
        """
        B, N, _ = transformer_out.shape
        ucg_sizes = self.ucg_sizes[:N]

        final = []
        for i, ucg_size in enumerate(ucg_sizes):
            final.append(
                transformer_out[:, i, :].view(B, -1, 3)[:, :ucg_size, :]
            )
        return torch.cat(final, dim=1)

    def forward(self, ucg_pos: Tensor, cg_disp: Tensor, t: Tensor) -> Tensor:
        """
        Forward pass of the Transformer backbone.

        Args:
            ucg_pos (Tensor): UCG positions.
            cg_disp (Tensor): Noisy CG displacements.
            t (Tensor): Diffusion timestep.

        Returns:
            Tensor: Predicted noise.
        """
        inputs = self._process_input(ucg_pos, cg_disp)
        inputs = self.embed_token(inputs)
        inputs = self.pos_encoder(inputs)
        
        src = torch.cat([self.embed_time(t), inputs], dim=1)
        out = self.transformer_encoder(src)
        out = self.lin_out(out)[:, 1:, :]
        return self._final_output(out)


####################### LightningModule #######################

class LitUCG2CGNoiseNet(L.LightningModule):
    """
    PyTorch Lightning Module for training and inference of the
    Transformer-based UCG-to-CG diffusion model.
    """

    def __init__(self, init_dim: int, dim: int, ff_dim: int, num_heads: int, num_layers: int, ucg_index_file: int, dropout: float, learn_rate: float):
        super().__init__()
        self.save_hyperparameters()

        # Read UCG sizes and scatter indices
        ucg_sizes = []
        data = np.load(ucg_index_file, allow_pickle=True)
        data = data["indices_per_cluster"]
        for i in range(data.shape[0]):
            ucg_sizes.append(len(data[i]))

        self.ucg_flat_idx = np.concatenate(data)
        num_cgs = len(self.ucg_flat_idx)
        self.scatter_idx = torch.zeros(num_cgs, dtype=torch.long)
        for i, indices in enumerate(data):
            self.scatter_idx[indices] = i
        self.scatter_idx, _ = self.scatter_idx.sort()

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

        # Parameters: learning rate
        self.learn_rate = learn_rate

    def forward(self, ucg_pos: Tensor, cg_disp: Tensor, t: Tensor) -> Tensor:
        return self.model(ucg_pos, cg_disp, t)

    def _get_loss(self, ucg_pos, cg_disp):
        """
        Computes MSE loss between predicted and true noise.
        """
        t = torch.empty(ucg_pos.size(0), 1, 1, device=cg_disp.device).uniform_(self.diffuser.t_min, self.diffuser.t_max)
        noisy_cg_disp, eps = self.diffuser.forward_noise(cg_disp, t)
        pred_eps = self.model(ucg_pos, noisy_cg_disp, t)
        return F.mse_loss(pred_eps, eps)

    def _get_combined_loss(self, combined_batch):
        """
        (Optional) Loss aggregation if using combined dataloaders.
        """
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
        """
        Configures optimizer (AdamW).
        """
        return torch.optim.AdamW(self.parameters(), lr=self.learn_rate)
    
    def generate(self, ucg_pos, num_steps):
        """
        Reverse diffusion process: generate CG beads from UCG beads.

        Args:
            ucg_pos (Tensor): UCG positions (B, 40, 3).
            num_steps (int): Number of diffusion steps.

        Returns:
            Tensor: Trajectory of generated CG bead positions over diffusion steps.
        """
        device  = ucg_pos.device
        B       = ucg_pos.shape[0]
        num_cgs = 751

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
