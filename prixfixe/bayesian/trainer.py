import json
import torch
import tqdm

import numpy as np
import torch.nn as nn

from pathlib import Path
from typing import Any

from ..prixfixe import Trainer, PrixFixeNet, DataProcessor, DEFAULT_METRICS


def setup_pyro_model(dataloader, pl_module):
    """Way to warmup Pyro Model and Guide in an automated way.

    Setup occurs before any device movement, so params are iniitalized on CPU.
    """
    from pyro import clear_param_store
    clear_param_store()
    for batch in dataloader:
        kwargs = {"dna_sequence": batch["x"].to(pl_module.device)}
        if "y_probs" in batch:  # classification
            kwargs["y_probs"] = batch["y_probs"].to(pl_module.device)
        else:  # regression
            kwargs["y"] = batch["y"].to(pl_module.device)
        pl_module.guide(**kwargs)
        pl_module.model(**kwargs)
        break


class BayesianTrainer(Trainer):
    def __init__(
        self,
        model: PrixFixeNet, 
        dataprocessor: DataProcessor,
        model_dir: str | Path,
        num_epochs: int,
        lr: float,
        device: torch.device = torch.device("cpu")):
        
        weight_decay = 0.01
        max_lr = lr
        # max_lr = self.deduce_max_lr()
        div_factor = 25.0
        min_lr = max_lr / div_factor
        model = model.to(device)
        super().__init__(model=model,
                         dataprocessor=dataprocessor,
                         model_dir=model_dir,
                         num_epochs=num_epochs,
                         device=device)
        
        optimizer = torch.optim.Adam(model.parameters(),
                                      lr = min_lr,)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, #type: ignore
                                                max_lr=max_lr,
                                                div_factor=div_factor,
                                                steps_per_epoch=dataprocessor.train_epoch_size(), 
                                                epochs=num_epochs, 
                                                pct_start=0.3,
                                                three_phase=False)
        self.optimizer=optimizer
        self.scheduler=scheduler

    def train_step(self, batch):   
        _, loss = self.model.train_step(batch)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        return loss.item()
    
    def on_epoch_end(self):
        """
        Autosome sheduler is called during training steps, not on each epoch end
        Nothing to do at epoch end 
        """
        pass
    
    def deduce_max_lr(self):
        # TODO: for now no solution to search for maximum lr automatically, learning rate range test should be analysed manually
        # MAX_LR=0.005 seems OK for most models 
        return 0.005
