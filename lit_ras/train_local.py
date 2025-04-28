"""
train_local.py

Local (single-node, single-GPU/CPU) training script for LitUCG2CGNoiseNet.

- Intended for small-scale training or debugging (e.g., on MacBooks with MPS, or single-GPU machines).
- Uses sample data from the local repository (`sample-data/`).
- Supports easy switching between MPS and GPU backends.
"""

####################### LC Hack: Workaround for CPU Affinity #######################
import os
import torch
import numpy as np

################################# Create lightning module and datamodule instances ####################################

from datamodules.ucg2cg import UCG2CGDataModule
from modules.diffusion_model import LitUCG2CGNoiseNet
from glob import glob

if __name__ == '__main__':

    # Datamodule for sample project data
    datamodule = UCG2CGDataModule(
        cg_files       = ["sample-data/cg/pfpatch_000000000138.npz", "sample-data/cg/pfpatch_000000000214.npz", "sample-data/cg/pfpatch_000000000272.npz"],
        ucg_files      = ["sample-data/ucg/pfpatch_000000000138_ucg.npz", "sample-data/ucg/pfpatch_000000000214_ucg.npz", "sample-data/ucg/pfpatch_000000000272_ucg.npz"],
        ucg_index_file = "sample-data/cg/all_indices_per_cluster.npz",
        batch_size     = 64,
        num_workers    = 1,
        train_size     = 0.9,
    )

    # Model definition (small config for faster testing)
    noise_net = LitUCG2CGNoiseNet(
        init_dim       = 120, # 39x3 + 1x3
        dim            = 64,
        ff_dim         = 32,
        num_heads      = 2,
        num_layers     = 1,
        ucg_index_file = "sample-data/cg/all_indices_per_cluster.npz",
        dropout        = 0.0,
        learn_rate     = 1e-4,
    )

    ################################# Start Training Session #####################################

    import pytorch_lightning as L
    from pytorch_lightning.callbacks import TQDMProgressBar
    from pytorch_lightning.loggers import TensorBoardLogger

    trainer = L.Trainer(
        max_steps = 100,                # Small number of steps for quick testing
        log_every_n_steps  = 5,         # Log every 5 steps
        val_check_interval = 1.0,       # Validate after every epoch
        strategy='auto',                # Automatic single-device strategy
        # accelerator='gpu',            # Uncomment if training on GPU
        accelerator='mps',              # Use MPS (Mac GPU backend) if available
        logger    = TensorBoardLogger(save_dir='./lit_logs/', name='dimers-C'),
        callbacks = [TQDMProgressBar(refresh_rate=1)],
    )

    # Example larger trainer config (commented out for production training)
    # trainer = L.Trainer(
    #     max_steps=1_000_000,
    #     log_every_n_steps=3000,
    #     val_check_interval=1.0,
    #     strategy='ddp', accelerator='gpu', devices=4, num_nodes=8,
    #     logger=TensorBoardLogger(save_dir='./lit_logs/', name='dimers-C'),
    #     callbacks=[TQDMProgressBar(refresh_rate=100)],
    # )

    print(f"Datamodule: {datamodule}")
    # print(f"Noise Net: {noise_net}")

    print("Starting Training")
    trainer.fit(
        noise_net, datamodule,
        # ckpt_path = ckpt_path # Uncomment to resume from checkpoint
    )
    print("Training Finished")
