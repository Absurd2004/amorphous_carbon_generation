"""
Flow Matching Transforms for Amorphous Carbon Generation.

These transforms implement the core Flow Matching logic:
1. FlowMatchingTransform: Interpolate between noise and data
2. DownselectEdges: Filter edges by cutoff (from DM2)

Key difference from DM2's RattleParticles (Denoising):
- Denoising: x_noisy = x_clean + σ * noise, predict noise
- Flow Matching: x_t = (1-t) * x_0 + t * x_1, predict velocity v = x_1 - x_0
"""

import torch
import numpy as np
from torch_geometric.transforms import BaseTransform


class FlowMatchingTransform(BaseTransform):
    """
    Flow Matching Transform for training.
    
    Given a clean sample x_1 (target), this transform:
    1. Samples a random initial point x_0 (from uniform or Gaussian prior)
    2. Samples a random time t ∈ [0, 1]
    3. Computes interpolated position: x_t = (1-t) * x_0 + t * x_1
    4. Computes target velocity: v = x_1 - x_0
    
    For 2D structures (z=0), the z-component is kept at 0.
    
    Args:
        prior: Prior distribution ('uniform' or 'gaussian')
        box_size: Box size for uniform prior [Lx, Ly, Lz]
        sigma: Standard deviation for Gaussian prior
        is_2d: Whether to enforce z=0 for 2D structures
        pbc: Whether to apply periodic boundary conditions
    """
    def __init__(
        self, 
        prior='uniform',
        box_size=[12.0, 12.0, 20.0],
        sigma=1.0,
        is_2d=True,
        pbc=True,
    ):
        super().__init__()
        self.prior = prior
        self.box_size = torch.tensor(box_size)
        self.sigma = sigma
        self.is_2d = is_2d
        self.pbc = pbc
        
    def __call__(self, data):
        """
        Apply Flow Matching transform to a data sample.
        
        Args:
            data: PyG Data object with:
                - pos: Clean atomic positions [num_atoms, 3] (x_1)
                - cell: Unit cell matrix (optional)
                
        Returns:
            data: Modified Data object with:
                - pos: Interpolated positions x_t [num_atoms, 3]
                - pos_target: Original clean positions x_1 [num_atoms, 3]
                - x_0: Initial random positions [num_atoms, 3]
                - t: Time parameter [1] or [batch_size]
                - velocity_target: Target velocity v = x_1 - x_0 [num_atoms, 3]
        """
        device = data.pos.device
        num_atoms = data.pos.size(0)
        box_size = self.box_size.to(device)
        
        # Store original clean positions as target
        x_1 = data.pos.clone()
        
        # Sample initial positions x_0 from prior
        if self.prior == 'uniform':
            x_0 = torch.rand(num_atoms, 3, device=device) * box_size
        elif self.prior == 'gaussian':
            # Gaussian centered at box center
            center = box_size / 2
            x_0 = torch.randn(num_atoms, 3, device=device) * self.sigma + center
            # Wrap into box if PBC
            if self.pbc:
                x_0 = x_0 % box_size
        else:
            raise ValueError(f"Unknown prior: {self.prior}")
        
        # Enforce 2D constraint
        if self.is_2d:
            x_0[:, 2] = 0.0
        
        # Sample time t ∈ [0, 1]
        t = torch.rand(1, device=device)
        
        # Compute target velocity v = x_1 - x_0
        # For PBC, use minimum image convention
        if self.pbc:
            velocity_target = self._min_image_diff(x_1, x_0, box_size)
        else:
            velocity_target = x_1 - x_0
        
        # Interpolate positions: x_t = x_0 + t * v = (1-t) * x_0 + t * x_1
        x_t = x_0 + t * velocity_target
        
        # Wrap into box if PBC
        if self.pbc:
            x_t = x_t % box_size
        
        # Update edge vectors for x_t positions
        if data.edge_attr is not None:
            i, j = data.edge_index
            if self.pbc:
                edge_vec = self._min_image_diff(x_t[j], x_t[i], box_size)
            else:
                edge_vec = x_t[j] - x_t[i]
            data.edge_attr = edge_vec
        
        # Store all quantities
        data.pos = x_t
        data.pos_target = x_1
        data.x_0 = x_0
        data.t = t
        data.velocity_target = velocity_target
        
        return data
    
    def _min_image_diff(self, x1, x0, box_size):
        """
        Compute minimum image difference under periodic boundary conditions.
        
        diff = x1 - x0, wrapped to [-L/2, L/2]
        """
        diff = x1 - x0
        diff = diff - box_size * torch.round(diff / box_size)
        return diff
    
    def forward(self, data):
        """Required for PyG transforms."""
        return self.__call__(data)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(prior={self.prior}, is_2d={self.is_2d})'


class FlowMatchingTransformBatch(BaseTransform):
    """
    Batched version of FlowMatchingTransform.
    
    For batched data, each sample in the batch gets a different time t.
    """
    def __init__(
        self, 
        prior='uniform',
        box_size=[12.0, 12.0, 20.0],
        sigma=1.0,
        is_2d=True,
        pbc=True,
    ):
        super().__init__()
        self.prior = prior
        self.box_size = torch.tensor(box_size)
        self.sigma = sigma
        self.is_2d = is_2d
        self.pbc = pbc
        
    def __call__(self, data):
        """
        Apply Flow Matching transform to a batched data sample.
        
        Args:
            data: Batched PyG Data object with:
                - pos: Clean atomic positions [total_atoms, 3]
                - batch: Batch assignment [total_atoms]
                
        Returns:
            data: Modified Data object with flow matching attributes
        """
        device = data.pos.device
        batch = data.batch
        num_graphs = batch.max().item() + 1
        box_size = self.box_size.to(device)
        
        # Store original clean positions
        x_1 = data.pos.clone()
        
        # Sample initial positions x_0 from prior
        if self.prior == 'uniform':
            x_0 = torch.rand_like(x_1) * box_size
        elif self.prior == 'gaussian':
            center = box_size / 2
            x_0 = torch.randn_like(x_1) * self.sigma + center
            if self.pbc:
                x_0 = x_0 % box_size
        else:
            raise ValueError(f"Unknown prior: {self.prior}")
        
        # Enforce 2D constraint
        if self.is_2d:
            x_0[:, 2] = 0.0
        
        # Sample time t ∈ [0, 1] for each graph in batch
        t = torch.rand(num_graphs, device=device)  # [num_graphs]
        t_nodes = t[batch]  # [total_atoms]
        
        # Compute target velocity
        if self.pbc:
            velocity_target = self._min_image_diff(x_1, x_0, box_size)
        else:
            velocity_target = x_1 - x_0
        
        # Interpolate positions: x_t = x_0 + t * v
        x_t = x_0 + t_nodes.unsqueeze(-1) * velocity_target
        
        # Wrap into box if PBC
        if self.pbc:
            x_t = x_t % box_size
        
        # Update edge vectors
        if data.edge_attr is not None:
            i, j = data.edge_index
            if self.pbc:
                edge_vec = self._min_image_diff(x_t[j], x_t[i], box_size)
            else:
                edge_vec = x_t[j] - x_t[i]
            data.edge_attr = edge_vec
        
        # Store all quantities
        data.pos = x_t
        data.pos_target = x_1
        data.x_0 = x_0
        data.t = t  # [num_graphs]
        data.velocity_target = velocity_target
        
        return data
    
    def _min_image_diff(self, x1, x0, box_size):
        diff = x1 - x0
        diff = diff - box_size * torch.round(diff / box_size)
        return diff
    
    def forward(self, data):
        return self.__call__(data)


class DownselectEdges(BaseTransform):
    """
    Downselect edges based on cutoff distance.
    
    Given a graph with edges (potentially constructed with a larger cutoff),
    filter to keep only edges shorter than the specified cutoff.
    
    This is useful when:
    1. Graph was constructed with a large cutoff initially
    2. After positions are perturbed, some edges may exceed cutoff
    
    From DM2/src/graphite/transforms/downselect_edges.py
    """
    def __init__(self, cutoff):
        super().__init__()
        self.cutoff = cutoff
    
    def __call__(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        
        # edge_attr contains displacement vectors [dx, dy, dz, ...]
        # Take the norm of the first 3 components
        edge_lengths = edge_attr[:, :3].norm(dim=1)
        mask = edge_lengths <= self.cutoff
        
        data.edge_index = edge_index[:, mask]
        data.edge_attr = edge_attr[mask]
        
        return data
    
    def forward(self, data):
        return self.__call__(data)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(cutoff={self.cutoff})'


class RebuildEdges(BaseTransform):
    """
    Rebuild edges after position update.
    
    Use ASE's primitive_neighbor_list for correct PBC handling.
    """
    def __init__(self, cutoff, pbc=True):
        super().__init__()
        self.cutoff = cutoff
        self.pbc = pbc
    
    def __call__(self, data):
        from ase.neighborlist import primitive_neighbor_list
        
        # Get positions and cell
        pos = data.pos.detach().cpu().numpy()
        cell = data.cell if hasattr(data, 'cell') else np.diag([12, 12, 20])
        pbc = [self.pbc, self.pbc, self.pbc]
        
        # Build neighbor list
        i, j, D = primitive_neighbor_list(
            'ijD', 
            cutoff=self.cutoff, 
            pbc=pbc,
            cell=cell, 
            positions=pos, 
            numbers=np.ones(len(pos))  # dummy
        )
        
        device = data.pos.device
        data.edge_index = torch.tensor(np.stack((i, j)), dtype=torch.long, device=device)
        data.edge_attr = torch.tensor(D, dtype=torch.float, device=device)
        
        return data
    
    def forward(self, data):
        return self.__call__(data)


def flow_matching_loss(pred_velocity, target_velocity, reduction='mean'):
    """
    Compute Flow Matching loss.
    
    Args:
        pred_velocity: Predicted velocity field [num_atoms, 3]
        target_velocity: Target velocity v = x_1 - x_0 [num_atoms, 3]
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        loss: MSE loss between predicted and target velocity
    """
    loss = torch.nn.functional.mse_loss(pred_velocity, target_velocity, reduction=reduction)
    return loss


if __name__ == "__main__":
    # Quick test
    from torch_geometric.data import Data
    
    print("Testing FlowMatchingTransform...")
    
    # Create dummy data
    num_atoms = 50
    pos = torch.rand(num_atoms, 3) * torch.tensor([12.0, 12.0, 0.0])  # 2D
    edge_index = torch.randint(0, num_atoms, (2, 100))
    edge_attr = torch.randn(100, 3)
    
    data = Data(pos=pos, edge_index=edge_index, edge_attr=edge_attr)
    
    # Apply transform
    transform = FlowMatchingTransform(
        prior='uniform',
        box_size=[12.0, 12.0, 20.0],
        is_2d=True,
        pbc=True,
    )
    
    data = transform(data)
    
    print(f"Original positions (x_1): {data.pos_target.shape}")
    print(f"Initial positions (x_0): {data.x_0.shape}")
    print(f"Interpolated positions (x_t): {data.pos.shape}")
    print(f"Time t: {data.t.item():.4f}")
    print(f"Velocity target shape: {data.velocity_target.shape}")
    print(f"Z coordinates (should be 0): {data.pos[:, 2].abs().max().item():.6f}")
    
    print("\nAll tests passed!")
