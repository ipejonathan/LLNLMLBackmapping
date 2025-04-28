"""
general.py

Defines diffusion processes used in the mini-MuMMI backmapping pipeline.

Implements both Variance-Preserving (VP) and Variance-Exploding (VE) diffusion schedules
for adding noise to CG displacement structures.
"""

import torch
import numpy as np
from torch import Tensor
from typing import Optional, Tuple, List

class VariancePreservingDiffuser:
    """
    Variance-Preserving (VP) Diffusion Process.

    Supports both 'cosine' and 'linear' noise schedules.  
    These schedules progressively add noise to inputs while preserving variance during training.

    Args:
        schedule (str): Type of noise schedule to use ('cosine' or 'linear').
        t_min (float): Minimum diffusion time (default 1e-3, start of diffusion).
        t_max (float): Maximum diffusion time (default 0.999, end of diffusion).
    """

    def __init__(self, schedule: str = 'cosine', t_min: float = 1e-3, t_max: float = 0.999) -> None:
        assert schedule in ['cosine', 'linear']

        self.t_min = t_min
        self.t_max = t_max
        self.schedule = schedule

        # Define diffusion coefficients based on schedule
        if schedule == 'cosine':
            # Alpha and sigma follow a cosine curve
            self.alpha = lambda t: torch.cos(torch.pi/2*t)
            self.sigma = lambda t: torch.sin(torch.pi/2*t)
            # Drift and diffusion coefficients
            self.f     = lambda t: torch.tan(torch.pi/2*t) * torch.pi * (-0.5)
            self.g2    = lambda t: torch.pi*self.alpha(t)*self.sigma(t) - 2*self.f(t)*(self.sigma(t)**2)
            self.g     = lambda t: self.g2(t)**0.5
        
        # Case 2: gamma = 1 - t, alpha = gamma.sqrt(), sigma = (1-gamma).sqrt()
        if schedule == 'linear':
            # Alpha and sigma vary linearly with t
            self.gamma = lambda t: 1 - t
            self.alpha = lambda t: self.gamma(t)**0.5
            self.sigma = lambda t: (1 - self.gamma(t))**0.5
            self.f     = lambda t: 0.5 / (t - 1)
            self.g2    = lambda t: 1 - 2*self.f(t)*t
            self.g     = lambda t: self.g2(t)**0.5

    def forward_noise(self, x: Tensor, t: Tensor, b: float = 1.0):
        """
        Applies one step of the forward diffusion process.

        Args:
            x (Tensor): Clean input tensor.
            t (Tensor): Diffusion timestep(s).
            b (float): Optional scaling factor for x.

        Returns:
            Tuple[Tensor, Tensor]: (noisy x, noise epsilon added)
        """
        alpha = self.alpha(t)
        sigma = self.sigma(t)
        eps = torch.randn_like(x)
        return alpha*x*b + sigma*eps, eps


class VarianceExplodingDiffuser:
    """
    Implements variance-exploding (VE) diffusion process.

    Forward process:
        Noise variance grows with time, starting from near-zero and exploding as t → t_max.

    Args:
        k (float): Scaling constant for variance growth.
        t_min (float): Minimum time value.
        t_max (float): Maximum time value.
    """
    def __init__(self, k: float  = 1.0, t_min: float = 1e-3, t_max: float = 0.999) -> None:
        self.t_min = t_min
        self.t_max = t_max

        # Define simple linear growth in noise
        self.alpha = lambda t: 1 # No decay of signal
        self.sigma = lambda t: k * t # Noise grows linearly
        self.f     = lambda t: 0
        self.g2    = lambda t: 2 * (k**2) * t
        self.g     = lambda t: self.g2(t)**0.5

    def forward_noise(self, x: Tensor, t: Tensor):
        """
        Applies one step of the forward diffusion process.

        Args:
            x (Tensor): Clean input tensor.
            t (Tensor): Diffusion timestep(s).

        Returns:
            Tuple[Tensor, Tensor]: (noisy x, noise epsilon added)
        """
        alpha = self.alpha(t)
        sigma = self.sigma(t)
        eps = torch.randn_like(x)
        return alpha * x + sigma * eps, eps
