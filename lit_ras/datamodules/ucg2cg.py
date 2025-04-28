"""
ucg2cg.py

Module for loading UCG and CG data for training and inference in mini-MuMMI's
machine learning-based backmapping pipeline.
"""

import torch
import numpy as np
from torch import Tensor
from typing import Tuple, List, Optional
from torch.utils.data import Dataset
import pytorch_lightning as L
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

####################### Datasets #######################

class UCG2CGDataset(Dataset):
    """
    PyTorch Dataset for UCG-to-CG backmapping tasks.

    Each sample contains:
        - UCG bead positions (centered at origin)
        - CG bead displacements relative to UCG beads
        - Mapping indices from CG beads to UCG beads

    Attributes:
        cg_files (List[str]): Paths to CG trajectory files (.npz).
        ucg_files (List[str]): Paths to UCG trajectory files (.npz).
        ucg_index_file (str): Path to file mapping CG beads to UCG clusters.
        scale (float): Scaling factor applied to input coordinates.
        scatter_idx (Tensor): Mapping tensor from CG beads to UCG beads.
    """

    def __init__(self, cg_files: List[str], ucg_files: List[str], ucg_index_file: str, scale: float = 1.0):
        super().__init__()
        self.cg_files       = cg_files
        self.ucg_files      = ucg_files
        self.ucg_index_file = ucg_index_file
        self.scale          = scale

        # Determine global-to-local indexing for map-style dataset implementation
        self.global_idx:   List[int]        = []  # global indices
        self.local_idx:    List[int]        = []  # local indices
        self.cg_pos_trajs: List[np.ndarray] = []  # CG position trajectories:
        self.ucg_pos_trajs: List[np.ndarray] = []

        # Load CG trajectories
        for i, fname in enumerate(cg_files):
            data = np.load(fname, allow_pickle=True)
            cg_pos_traj = data['protein_positions']
            traj_len = len(cg_pos_traj)
            self.global_idx.extend([i]*traj_len)
            self.local_idx.extend(list(range(traj_len)))
            self.cg_pos_trajs.append(cg_pos_traj)

        # Load UCG trajectories
        for i, fname in enumerate(ucg_files):
            data = np.load(fname, allow_pickle=True)
            ucg_pos_traj = data['positions_ucg']
            self.ucg_pos_trajs.append(ucg_pos_traj)

        # Read UCG index file
        data = np.load(ucg_index_file, allow_pickle=True)
        self.ucg_idx = data["indices_per_cluster"]

        # Flatten UCG indices
        self.ucg_flat_idx = np.concatenate(self.ucg_idx)

        # Get scatter indices from ucg_idx
        num_cgs = len(self.ucg_flat_idx)
        self.scatter_idx = torch.zeros(num_cgs, dtype=torch.long)
        for i, indices in enumerate(self.ucg_idx):
            self.scatter_idx[indices] = i

        # Reorder scatter_idx
        self.scatter_idx, _ = self.scatter_idx.sort()

    def __len__(self):
        """Returns the number of available samples."""
        return len(self.local_idx)
    
    def __getitem__(self, idx):
        """
        Retrieves one data sample.

        Args:
            idx (int): Sample index.

        Returns:
            dict: Contains UCG positions, CG displacements, scatter indices, centered CG positions, and UCG origin.
        """
        global_idx = self.global_idx[idx]
        local_idx  = self.local_idx[idx]

        # Load CG position
        cg_pos = self.cg_pos_trajs[global_idx][local_idx] * self.scale
        cg_pos = torch.tensor(cg_pos, dtype=torch.float)

        # Reorder CG beads to match UCG layout
        cg_pos = cg_pos[self.ucg_flat_idx]

        # Load UCG position
        ucg_pos = self.ucg_pos_trajs[global_idx][local_idx] * self.scale
        ucg_pos = torch.tensor(ucg_pos, dtype=torch.float)

        # Compute displacements between CG and UCG
        cg_disp = cg_pos - ucg_pos[self.scatter_idx]

        # Center UCG around origin
        origin = ucg_pos.mean(dim=0, keepdim=True)
        ucg_pos -= origin

        return {
            'ucg_pos': ucg_pos,
            'cg_disp': cg_disp,
            'scatter_idx': self.scatter_idx,
            'cg_pos': ucg_pos[self.scatter_idx] + cg_disp,
            'origin': origin,
        }

####################### LightningDataModules #######################

class UCG2CGDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule to manage UCG2CG datasets.

    Handles splitting into train/validation sets, DataLoader creation.

    Args:
        cg_files (List[str]): List of CG trajectory .npz files.
        ucg_files (List[str]): List of UCG trajectory .npz files.
        ucg_index_file (str): Path to the UCG index .npz file.
        scale (float): Optional scaling factor for positions.
        batch_size (int): Batch size for training/validation.
        num_workers (int): Number of DataLoader workers.
        train_size (float): Fraction of data to use for training.
    """
    def __init__(self, cg_files: List[str], ucg_files: List[str], ucg_index_file: str,
                 scale: float = 1.0, batch_size: int = 64, num_workers: int = 4, train_size: float = 0.99):
        super().__init__()
        self.save_hyperparameters(ignore='cg_files')

        self.cg_files       = cg_files
        self.ucg_files      = ucg_files
        self.ucg_index_file = ucg_index_file
        self.scale          = scale
        self.batch_size     = batch_size
        self.num_workers    = num_workers
        self.train_size     = train_size

    def prepare_data(self):
        """Prepare data before training. (Not used here.)"""
        # Download, IO, etc. Useful with shared filesystems
        # Only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage: Optional[str] = None):
        """Split data into training and validation sets."""

        self.cg_train_files, self.cg_valid_files, self.ucg_train_files, self.ucg_valid_files = train_test_split(
            self.cg_files, self.ucg_files, train_size=self.train_size, random_state=42
        )

        self.train_set = []
        if stage in ['fit', 'train', None]:
            self.train_set = UCG2CGDataset(self.cg_train_files, self.ucg_train_files, self.ucg_index_file, self.scale)
        
        self.valid_set = UCG2CGDataset(self.cg_valid_files, self.ucg_valid_files, self.ucg_index_file, self.scale)

    def train_dataloader(self):
        """Returns DataLoader for training."""
        return DataLoader(self.train_set, shuffle=True,  batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True)
    
    def val_dataloader(self):
        """Returns DataLoader for validation."""
        return DataLoader(self.valid_set, shuffle=False, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True)

    def teardown(self, stage: Optional[str] = None):
        """Optional cleanup at end of training."""
        pass