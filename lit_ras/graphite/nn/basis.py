"""
basis.py

Implements scalar-to-basis function expansions for neural networks:
- Bessel basis
- Gaussian basis
- Gaussian Random Fourier Features

Used for expanding scalar inputs (like distances or scalar features) into richer embeddings
for input to equivariant neural networks or diffusion models.
"""

import torch
from torch import nn
from torch import Tensor
from typing import List, Optional, Tuple

####################### Basis Function Utilities #######################

def bessel(x: Tensor, start: float = 0.0, end: float = 1.0, num_basis: int = 8, eps: float = 1e-5) -> Tensor:
    """
    Expand scalar input into radial Bessel basis functions.

    Args:
        x (Tensor): Input tensor of scalars.
        start (float): Start of the range.
        end (float): End of the range.
        num_basis (int): Number of basis functions.
        eps (float): Small epsilon to avoid division by zero.

    Returns:
        Tensor: Bessel basis expansion of x, shape (..., num_basis).
    """
    x = x[..., None] - start + eps
    c = end - start
    n = torch.arange(1, num_basis+1, dtype=x.dtype, device=x.device)
    return ((2/c)**0.5) * torch.sin(n * torch.pi * x / c) / x


def gaussian(x: Tensor, start: float = 0.0, end: float = 1.0, num_basis: int = 8) -> Tensor:
    """
    Expand scalar input into Gaussian basis functions.

    Args:
        x (Tensor): Input tensor of scalars.
        start (float): Start of the range.
        end (float): End of the range.
        num_basis (int): Number of basis functions.

    Returns:
        Tensor: Gaussian basis expansion of x, shape (..., num_basis).
    """
    mu = torch.linspace(start, end, num_basis, dtype=x.dtype, device=x.device)
    step = mu[1] - mu[0]
    diff = (x[..., None] - mu) / step
    return diff.pow(2).neg().exp().div(1.12) # Normalized so sum of squares ≈ 1


def scalar2basis(x: Tensor, start: float, end: float, num_basis: int, basis: str = 'gaussian'):
    """
    Expand scalar input using specified basis functions.

    Args:
        x (Tensor): Input tensor.
        start (float): Start of the range.
        end (float): End of the range.
        num_basis (int): Number of basis functions.
        basis (str): 'gaussian' or 'bessel'.

    Returns:
        Tensor: Basis-expanded input.

    Reference:
        https://docs.e3nn.org/en/stable/api/math/math.html#e3nn.math.soft_one_hot_linspace.
    """
    funcs = {
        'gaussian': gaussian,
        'bessel': bessel,
    }
    return funcs[basis](x, start, end, num_basis)

####################### Basis Function Modules #######################

class Bessel(nn.Module):
    """
    Bessel basis expansion module for neural networks.

    Args:
        start (float): Start of the input range.
        end (float): End of the input range.
        num_basis (int): Number of basis functions.
        eps (float): Small value to avoid division by zero.
    """

    def __init__(self, start: float = 0.0, end: float = 1.0, num_basis: int = 8, eps: float = 1e-5) -> None:
        super().__init__()
        self.start     = start
        self.end       = end
        self.num_basis = num_basis
        self.eps       = eps
        self.register_buffer('n', torch.arange(1, num_basis+1, dtype=torch.float))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for Bessel basis expansion.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Basis-expanded tensor.
        """
        x = x[..., None] - self.start + self.eps
        c = self.end - self.start
        return ((2/c)**0.5) * torch.sin(self.n*torch.pi*x / c) / x

    def extra_repr(self) -> str:
        return f'start={self.start}, end={self.end}, num_basis={self.num_basis}, eps={self.eps}'


class Gaussian(nn.Module):
    """
    Gaussian basis expansion module for neural networks.

    Args:
        start (float): Start of the input range.
        end (float): End of the input range.
        num_basis (int): Number of basis functions.
    """

    def __init__(self, start: float = 0.0, end: float = 1.0, num_basis: int = 8) -> None:
        super().__init__()
        self.start     = start
        self.end       = end
        self.num_basis = num_basis
        self.register_buffer('mu', torch.linspace(start, end, num_basis))
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for Gaussian basis expansion.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Basis-expanded tensor.
        """
        step = self.mu[1] - self.mu[0]
        diff = (x[..., None] - self.mu) / step
        return diff.pow(2).neg().exp().div(1.12) # division by 1.12 so that sum of square is roughly 1

    def extra_repr(self) -> str:
        return f'start={self.start}, end={self.end}, num_basis={self.num_basis}'


class GaussianRandomFourierFeatures(nn.Module):
    """
    Gaussian Random Fourier Features (RFF) for positional encoding.

    Projects input into a higher-dimensional space using random Gaussian matrix.

    Reference:
        "Gaussian Fourier Features for Neural Networks" (https://arxiv.org/abs/2006.10739)

    Args:
        embed_dim (int): Target embedding dimension.
        input_dim (int): Input dimension.
        sigma (float): Standard deviation for random Gaussian weights.
    """

    def __init__(self, embed_dim: int, input_dim: int = 1, sigma: float = 1.0) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.sigma     = sigma

        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.register_buffer('B', torch.randn(input_dim, embed_dim//2) * sigma)

    def forward(self, v: Tensor) -> Tensor:
        """
        Forward pass for Fourier feature projection.

        Args:
            v (Tensor): Input tensor.

        Returns:
            Tensor: Concatenation of cosine and sine projections.
        """
        v_proj =  2 * torch.pi * v @ self.B
        return torch.cat([torch.cos(v_proj), torch.sin(v_proj)], dim=-1)

    def extra_repr(self) -> str:
            return f'embed_dim={self.embed_dim}, input_dim={self.input_dim}, sigma={self.sigma}'
