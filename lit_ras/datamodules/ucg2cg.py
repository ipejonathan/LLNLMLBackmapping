import torch
import numpy as np

# Typing
from torch import Tensor
from typing import Tuple, List, Optional

####################### Datasets #######################

from torch.utils.data import Dataset
# from torch_scatter import scatter

class UCG2CGDataset(Dataset):
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
        return len(self.local_idx)
    
    def __getitem__(self, idx):
        global_idx = self.global_idx[idx]
        local_idx  = self.local_idx[idx]

        # Extract CG pos
        cg_pos = self.cg_pos_trajs[global_idx][local_idx] * self.scale
        cg_pos = torch.tensor(cg_pos, dtype=torch.float)

        # Reorder CG pos
        cg_pos = cg_pos[self.ucg_flat_idx]

        # Compute UCG pos
        ucg_pos = self.ucg_pos_trajs[global_idx][local_idx] * self.scale
        ucg_pos = torch.tensor(ucg_pos, dtype=torch.float)

        # Compute CG displacements
        cg_disp = cg_pos - ucg_pos[self.scatter_idx]

        # Center UCG pos
        origin = ucg_pos.mean(dim=0, keepdim=True)
        ucg_pos -= origin

        return {
            'ucg_pos': ucg_pos,
            'cg_disp': cg_disp,
            'cg_pos': ucg_pos[self.scatter_idx] + cg_disp,
            'origin': origin,
        }


####################### LightningDataModules #######################

import pytorch_lightning as L
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
# from pytorch_lightning.pytorch.utilities import CombinedLoader

class UCG2CGDataModule(L.LightningDataModule):
    def __init__(self, cg_files: List[str], ucg_files: List[str], ucg_index_file: str, scale: float = 1.0, batch_size: int = 64, num_workers: int = 4, train_size: float = 0.99):
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
        # Download, IO, etc. Useful with shared filesystems
        # Only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage: Optional[str] = None):
        # Make assignments here (val/train/test split)
        # Called on every process in DDP
        self.cg_train_files, self.cg_valid_files, self.ucg_train_files, self.ucg_valid_files = train_test_split(self.cg_files, self.ucg_files, train_size=self.train_size, random_state=42)

        # Zip the paired lists together before splitting
        # paired_files = list(zip(self.cg_files, self.ucg_files))
        # # Split into train and validation sets
        # train_pairs, valid_pairs = train_test_split(paired_files, train_size=self.train_size, random_state=42)
        # # Unzip the pairs back into separate lists
        # self.cg_train_files, self.ucg_train_files = zip(*train_pairs)
        # self.cg_valid_files, self.ucg_valid_files = zip(*valid_pairs)

        self.train_set = []
        if stage in ['fit', 'train', None]:
            self.train_set = UCG2CGDataset(self.cg_train_files, self.ucg_train_files, self.ucg_index_file, self.scale)
        
        self.valid_set = UCG2CGDataset(self.cg_valid_files, self.ucg_valid_files, self.ucg_index_file, self.scale)

    def train_dataloader(self):
        return DataLoader(self.train_set, shuffle=True,  batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.valid_set, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def teardown(self, stage: Optional[str] = None):
        # Clean up state after the trainer stops, delete files...
        # Called on every process in DDP
        pass



datamodule = UCG2CGDataModule(
    cg_files       = ["sample-data/cg/pfpatch_000000000138.npz", "sample-data/cg/pfpatch_000000000214.npz", "sample-data/cg/pfpatch_000000000272.npz"],
    ucg_files      = ["sample-data/ucg/pfpatch_000000000138_ucg.npz", "sample-data/ucg/pfpatch_000000000214_ucg.npz", "sample-data/ucg/pfpatch_000000000272_ucg.npz"],
    ucg_index_file = "sample-data/cg/all_indices_per_cluster.npz",
    batch_size     = 64,
    num_workers    = 8,
    train_size     = 0.9,
)
datamodule.setup()

# datamodule.train_dataloader()
# print(datamodule)


# cg_files = ["/Users/jonathan/Documents/LLNLMLBackmapping/sample-data/cg/pfpatch_000000000138.npz", "/Users/jonathan/Documents/LLNLMLBackmapping/sample-data/cg/pfpatch_000000000214.npz", "/Users/jonathan/Documents/LLNLMLBackmapping/sample-data/cg/pfpatch_000000000272.npz"]
# ucg_idx_file = "/Users/jonathan/Documents/LLNLMLBackmapping/sample-data/cg/all_indices_per_cluster.npz"
# ucg_files = ["/Users/jonathan/Documents/LLNLMLBackmapping/sample-data/ucg/pfpatch_000000000138_ucg.npz", "/Users/jonathan/Documents/LLNLMLBackmapping/sample-data/ucg/pfpatch_000000000214_ucg.npz", "/Users/jonathan/Documents/LLNLMLBackmapping/sample-data/ucg/pfpatch_000000000272_ucg.npz"]
# dataset = UCG2CGDataset(cg_files=cg_files, ucg_files=ucg_files, ucg_index_file=ucg_idx_file)

# print(dataset)
# print(len(dataset))
# print(dataset[0]['ucg_pos'])
# print(dataset[0]['cg_pos'])