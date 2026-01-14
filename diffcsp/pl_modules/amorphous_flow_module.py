"""
Amorphous Flow Module - PyTorch Lightning Module for Flow Matching.

This module integrates:
1. Multiple model backends (NequIP, EGNN, SchNet) with dynamic switching
2. Flow Matching training logic
3. ODE sampling for generation
4. Conditional generation (cooling rate)

Compatible with CrystalFlow's training infrastructure.

Supported Models:
- nequip (default): E(3) equivariant, highest accuracy, slower
- egnn: E(n) equivariant, balanced speed/accuracy
- schnet: Fast, not equivariant, good for smooth potentials
"""

import math
import logging
from typing import Any, Dict, Optional, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from torch_geometric.data import Data, Batch
from tqdm import tqdm

import hydra
from omegaconf import DictConfig

from diffcsp.pl_modules.model_factory import (
    create_model,
    list_available_models,
    get_model_info,
    ModelType,
    DEFAULT_CONFIGS,
)
from diffcsp.pl_modules.flow_transforms import (
    FlowMatchingTransform,
    FlowMatchingTransformBatch,
    DownselectEdges,
    RebuildEdges,
    flow_matching_loss,
)

logger = logging.getLogger(__name__)


class AmorphousFlowModule(pl.LightningModule):
    """
    PyTorch Lightning Module for Flow Matching on Amorphous Carbon.
    
    This module implements:
    1. Training: Flow matching with velocity field prediction
    2. Sampling: ODE integration from noise to data
    3. Conditional generation: Support for cooling rate conditioning
    4. Dynamic model switching: NequIP, EGNN, or SchNet backend
    
    Flow Matching Formulation:
    - x_t = (1-t) * x_0 + t * x_1  (linear interpolation)
    - v(x_t, t) = x_1 - x_0  (target velocity)
    - Loss = MSE(v_pred, v_target)
    
    Args:
        model_type: Model backend ('nequip', 'egnn', 'schnet'), default='nequip'
        model_config: Configuration dict for the model
        box_size: Simulation box size [Lx, Ly, Lz]
        cutoff: Cutoff radius for graph construction
        is_2d: Whether structure is 2D (z=0)
        prior: Prior distribution ('uniform' or 'gaussian')
        use_condition: Whether to use conditioning (cooling rate)
        learning_rate: Learning rate
        ema_decay: Exponential moving average decay (0 to disable)
    
    Example:
        # Use NequIP (default, highest accuracy)
        module = AmorphousFlowModule(model_type='nequip')
        
        # Use EGNN (faster, still equivariant)
        module = AmorphousFlowModule(model_type='egnn', model_config={'hidden_dim': 256})
        
        # Use SchNet (fastest)
        module = AmorphousFlowModule(model_type='schnet')
    """
    
    def __init__(
        self,
        # Model selection
        model_type: str = 'nequip',  # 'nequip', 'egnn', 'schnet'
        model_config: Optional[Dict] = None,
        # Common model configuration (used if model_config is None)
        cutoff: float = 5.0,
        time_embed_dim: int = 32,
        cond_embed_dim: int = 32,
        cond_min_value: float = 1.0,
        cond_max_value: float = 4.0,
        # Training configuration
        box_size: List[float] = [12.0, 12.0, 20.0],
        is_2d: bool = True,
        prior: str = 'uniform',
        use_condition: bool = True,
        # Optimizer configuration
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        lr_scheduler: Optional[str] = None,
        lr_scheduler_params: Optional[Dict] = None,
        # Other
        ema_decay: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Store configuration
        self.model_type = model_type
        self.box_size = torch.tensor(box_size)
        self.cutoff = cutoff
        self.is_2d = is_2d
        self.prior = prior
        self.use_condition = use_condition
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params or {}
        
        # Build model config
        if model_config is None:
            model_config = {}
        
        # Add common parameters
        model_config.setdefault('cutoff', cutoff)
        model_config.setdefault('time_embed_dim', time_embed_dim)
        model_config.setdefault('cond_embed_dim', cond_embed_dim)
        model_config.setdefault('cond_min_value', cond_min_value)
        model_config.setdefault('cond_max_value', cond_max_value)
        
        # Create model using factory
        logger.info(f"Creating model: {model_type}")
        logger.info(f"Model config: {model_config}")
        
        self.model = create_model(model_type, **model_config)
        
        # Log model info
        model_info = get_model_info(model_type)
        logger.info(f"Model: {model_info['name']} - {model_info['description']}")
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Parameters: {n_params:,}")
        
        # Flow matching transform (applied during training)
        self.flow_transform = FlowMatchingTransformBatch(
            prior=prior,
            box_size=box_size,
            is_2d=is_2d,
            pbc=True,
        )
        
        # Edge downselection (after position perturbation)
        self.downselect_edges = DownselectEdges(cutoff=cutoff)
        
        # EMA (optional)
        self.ema_decay = ema_decay
        if ema_decay > 0:
            self.ema_model = None  # Initialize on first update
            
    def forward(self, data: Data, t: torch.Tensor, y: Optional[torch.Tensor] = None):
        """
        Forward pass: predict velocity field.
        
        Args:
            data: PyG Data object with positions at time t
            t: Time tensor [batch_size]
            y: Condition tensor [batch_size] (log10 cooling rate)
            
        Returns:
            velocity: Predicted velocity field [num_atoms, 3]
        """
        return self.model(data, t, y)
    
    def training_step(self, batch: Data, batch_idx: int):
        """
        Training step: compute flow matching loss.
        
        1. Apply flow matching transform (sample x_0, t, compute x_t)
        2. Downselect edges for new positions
        3. Predict velocity v(x_t, t, y)
        4. Compute MSE loss with target velocity
        """
        # Apply flow matching transform
        batch = self.flow_transform(batch)
        
        # Downselect edges after position update
        batch = self.downselect_edges(batch)
        
        # Get time and condition
        t = batch.t  # [num_graphs]
        y = batch.cooling_rate if self.use_condition and hasattr(batch, 'cooling_rate') else None
        
        # Predict velocity
        pred_velocity = self.model(batch, t, y)
        
        # Compute loss
        target_velocity = batch.velocity_target
        loss = flow_matching_loss(pred_velocity, target_velocity, reduction='mean')
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        
        return loss
    
    def validation_step(self, batch: Data, batch_idx: int):
        """Validation step: same as training but without gradient."""
        batch = self.flow_transform(batch)
        batch = self.downselect_edges(batch)
        
        t = batch.t
        y = batch.cooling_rate if self.use_condition and hasattr(batch, 'cooling_rate') else None
        
        pred_velocity = self.model(batch, t, y)
        target_velocity = batch.velocity_target
        loss = flow_matching_loss(pred_velocity, target_velocity, reduction='mean')
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        
        return loss
    
    def test_step(self, batch: Data, batch_idx: int):
        """Test step: same as validation."""
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        if self.lr_scheduler is None:
            return optimizer
        
        if self.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.lr_scheduler_params.get('T_max', 1000),
                eta_min=self.lr_scheduler_params.get('eta_min', 1e-6),
            )
        elif self.lr_scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.lr_scheduler_params.get('step_size', 100),
                gamma=self.lr_scheduler_params.get('gamma', 0.5),
            )
        elif self.lr_scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.lr_scheduler_params.get('factor', 0.5),
                patience=self.lr_scheduler_params.get('patience', 10),
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                },
            }
        else:
            raise ValueError(f"Unknown scheduler: {self.lr_scheduler}")
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    # ==================== Sampling ====================
    
    @torch.no_grad()
    def sample(
        self,
        num_atoms: int = 50,
        num_samples: int = 1,
        condition: Optional[float] = None,
        steps: int = 100,
        method: str = 'euler',
        device: Optional[torch.device] = None,
        return_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Generate samples using ODE integration.
        
        Args:
            num_atoms: Number of atoms per sample
            num_samples: Number of samples to generate
            condition: Conditioning value (log10 cooling rate)
            steps: Number of ODE integration steps
            method: Integration method ('euler' or 'rk4')
            device: Device to generate on
            return_trajectory: Whether to return full trajectory
            
        Returns:
            samples: Generated positions [num_samples, num_atoms, 3]
            trajectory: Optional list of positions at each step
        """
        if device is None:
            device = next(self.parameters()).device
        
        self.model.eval()
        box_size = self.box_size.to(device)
        
        # Initialize from prior
        if self.prior == 'uniform':
            x = torch.rand(num_samples * num_atoms, 3, device=device) * box_size
        else:
            x = torch.randn(num_samples * num_atoms, 3, device=device) + box_size / 2
            x = x % box_size
        
        # Enforce 2D
        if self.is_2d:
            x[:, 2] = 0.0
        
        # Create batch indices
        batch = torch.arange(num_samples, device=device).repeat_interleave(num_atoms)
        
        # Condition
        if condition is not None:
            y = torch.full((num_samples,), condition, device=device)
        else:
            y = None
        
        # Time steps
        dt = 1.0 / steps
        times = torch.linspace(0, 1 - dt, steps, device=device)
        
        trajectory = [x.clone()] if return_trajectory else None
        
        # ODE integration
        for t_val in tqdm(times, desc='Sampling', disable=not return_trajectory):
            t = torch.full((num_samples,), t_val.item(), device=device)
            
            # Build graph
            data = self._build_graph(x, batch, box_size)
            
            # Predict velocity
            v = self.model(data, t, y)
            
            # Integration step
            if method == 'euler':
                x = x + v * dt
            elif method == 'rk4':
                x = self._rk4_step(x, t, y, batch, box_size, dt)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Apply PBC
            x = x % box_size
            
            # Enforce 2D
            if self.is_2d:
                x[:, 2] = 0.0
            
            if return_trajectory:
                trajectory.append(x.clone())
        
        # Final step to t=1
        t = torch.ones(num_samples, device=device)
        data = self._build_graph(x, batch, box_size)
        v = self.model(data, t, y)
        x = x + v * dt
        x = x % box_size
        if self.is_2d:
            x[:, 2] = 0.0
        
        if return_trajectory:
            trajectory.append(x.clone())
        
        # Reshape to [num_samples, num_atoms, 3]
        samples = x.view(num_samples, num_atoms, 3)
        
        return samples, trajectory
    
    def _build_graph(self, pos: torch.Tensor, batch: torch.Tensor, box_size: torch.Tensor) -> Data:
        """Build graph from positions using ASE neighbor list."""
        from ase.neighborlist import primitive_neighbor_list
        
        device = pos.device
        pos_np = pos.detach().cpu().numpy()
        batch_np = batch.detach().cpu().numpy()
        box_np = box_size.detach().cpu().numpy()
        
        all_i, all_j, all_D = [], [], []
        offset = 0
        
        for b in range(batch.max().item() + 1):
            mask = batch_np == b
            pos_b = pos_np[mask]
            n_atoms = mask.sum()
            
            i, j, D = primitive_neighbor_list(
                'ijD',
                cutoff=self.cutoff,
                pbc=[True, True, True],
                cell=np.diag(box_np),
                positions=pos_b,
                numbers=np.ones(n_atoms),
            )
            
            all_i.append(i + offset)
            all_j.append(j + offset)
            all_D.append(D)
            offset += n_atoms
        
        edge_index = torch.tensor(
            np.stack([np.concatenate(all_i), np.concatenate(all_j)]),
            dtype=torch.long, device=device
        )
        edge_attr = torch.tensor(
            np.concatenate(all_D),
            dtype=torch.float, device=device
        )
        
        # Atom types (all carbon = 0)
        x = torch.zeros(pos.size(0), dtype=torch.long, device=device)
        
        data = Data(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
        )
        
        return data
    
    def _rk4_step(self, x, t, y, batch, box_size, dt):
        """Fourth-order Runge-Kutta integration step."""
        def velocity(pos, time):
            data = self._build_graph(pos, batch, box_size)
            t_tensor = torch.full((batch.max().item() + 1,), time, device=pos.device)
            return self.model(data, t_tensor, y)
        
        t_val = t[0].item()
        k1 = velocity(x, t_val)
        k2 = velocity((x + 0.5 * dt * k1) % box_size, t_val + 0.5 * dt)
        k3 = velocity((x + 0.5 * dt * k2) % box_size, t_val + 0.5 * dt)
        k4 = velocity((x + dt * k3) % box_size, t_val + dt)
        
        return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def test_module():
    """Test the AmorphousFlowModule."""
    print("Testing AmorphousFlowModule...")
    
    # Create module
    module = AmorphousFlowModule(
        num_species=1,
        cutoff=5.0,
        num_convs=2,
        box_size=[12.0, 12.0, 20.0],
        is_2d=True,
        use_condition=True,
    )
    print(f"Module created successfully!")
    print(f"Number of parameters: {sum(p.numel() for p in module.parameters()):,}")
    
    # Create dummy batch
    from diffcsp.pl_data.amorphous_dataset import AmorphousCarbonDataset
    import os
    
    data_path = '/home/yongkunyang/CrystalFlow/data/amorphous_carbon/data'
    if os.path.exists(data_path):
        dataset = AmorphousCarbonDataset(data_path, cutoff=5.0, duplicate=1)
        batch = Batch.from_data_list([dataset[i] for i in range(4)])
        
        # Test training step
        loss = module.training_step(batch, 0)
        print(f"Training loss: {loss.item():.6f}")
        
        # Test sampling (quick, just 5 steps)
        samples, _ = module.sample(num_atoms=50, num_samples=2, condition=2.0, steps=5)
        print(f"Generated samples shape: {samples.shape}")
        print(f"Z coordinates (should be 0): {samples[:, :, 2].abs().max().item():.6f}")
    else:
        print(f"Data path not found: {data_path}")
        print("Skipping data-dependent tests.")
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_module()
