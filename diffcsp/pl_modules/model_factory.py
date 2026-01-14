"""
Model Factory for Amorphous Flow Matching.

提供多种 GNN 模型的统一接口，支持动态切换：
1. NequIP (默认) - E3 等变神经网络，精度高
2. EGNN - E(n) 等变网络，轻量级
3. SchNet - 基于连续滤波的网络

使用方式:
    model = create_model('nequip', **config)
    model = create_model('egnn', **config)  
    model = create_model('schnet', **config)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import math

from torch_geometric.data import Data, Batch


# ============================================================================
# Model Types
# ============================================================================

class ModelType(Enum):
    """Supported model types."""
    NEQUIP = "nequip"
    EGNN = "egnn"
    SCHNET = "schnet"
    
    @classmethod
    def from_string(cls, s: str) -> "ModelType":
        """Convert string to ModelType."""
        s = s.lower()
        for mt in cls:
            if mt.value == s:
                return mt
        raise ValueError(f"Unknown model type: {s}. Supported: {[m.value for m in cls]}")


# ============================================================================
# Common Components
# ============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding (same as positional encoding in Transformers)."""
    
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
    
    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: (B,) or (B, 1) time tensor
            
        Returns:
            (B, dim) embeddings
        """
        if t.dim() == 2:
            t = t.squeeze(-1)
        
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * 
            torch.arange(half_dim, device=t.device, dtype=t.dtype) / half_dim
        )
        args = t[:, None] * freqs[None, :]
        
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1), value=0)
        
        return embedding


class GaussianSmearing(nn.Module):
    """Gaussian smearing of distances."""
    
    def __init__(self, start: float = 0.0, stop: float = 5.0, num_gaussians: int = 50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.register_buffer('offset', offset)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
    
    def forward(self, dist: Tensor) -> Tensor:
        """
        Args:
            dist: (E,) distances
            
        Returns:
            (E, num_gaussians) smeared distances
        """
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * dist ** 2)


class MLP(nn.Module):
    """Multi-layer perceptron with optional residual connection."""
    
    def __init__(
        self,
        dims: List[int],
        act: nn.Module = nn.SiLU(),
        residual: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.residual = residual
        
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        self.net = nn.Sequential(*layers)
        
        if residual and dims[0] != dims[-1]:
            self.shortcut = nn.Linear(dims[0], dims[-1])
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: Tensor) -> Tensor:
        out = self.net(x)
        if self.residual:
            out = out + self.shortcut(x)
        return out


def scatter(
    src: Tensor,
    index: Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None,
    reduce: str = 'sum',
) -> Tensor:
    """Scatter operation without torch_scatter dependency."""
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    
    shape = list(src.shape)
    shape[dim] = dim_size
    
    if reduce in ('sum', 'add'):
        out = torch.zeros(shape, dtype=src.dtype, device=src.device)
        return out.scatter_add(dim, index.unsqueeze(-1).expand_as(src), src)
    
    elif reduce == 'mean':
        out = torch.zeros(shape, dtype=src.dtype, device=src.device)
        out = out.scatter_add(dim, index.unsqueeze(-1).expand_as(src), src)
        ones = torch.ones_like(src)
        count = torch.zeros(shape, dtype=src.dtype, device=src.device)
        count = count.scatter_add(dim, index.unsqueeze(-1).expand_as(src), ones)
        count = count.clamp(min=1)
        return out / count
    
    else:
        raise ValueError(f"Unknown reduce operation: {reduce}")


# ============================================================================
# Base Model Interface
# ============================================================================

class FlowMatchingModel(nn.Module):
    """Base class for flow matching models."""
    
    def forward(
        self, 
        data: Data, 
        t: Tensor, 
        y: Optional[Tensor] = None
    ) -> Tensor:
        """
        Predict velocity field.
        
        Args:
            data: PyG Data with pos, edge_index, edge_attr, batch
            t: (B,) time tensor
            y: (B,) condition tensor (optional)
            
        Returns:
            (N, 3) predicted velocity
        """
        raise NotImplementedError


# ============================================================================
# EGNN Model (Lightweight Equivariant)
# ============================================================================

class EGNNFlowMatching(FlowMatchingModel):
    """
    E(n) Equivariant Graph Neural Network for Flow Matching.
    
    Lightweight alternative to NequIP, updates both features and positions.
    Based on: E(n) Equivariant Graph Neural Networks (Satorras et al., 2021)
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_gaussians: int = 50,
        cutoff: float = 5.0,
        time_embed_dim: int = 64,
        cond_embed_dim: int = 64,
        cond_min_value: float = 1.0,
        cond_max_value: float = 4.0,
        update_pos: bool = False,  # For velocity prediction, don't update pos
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff
        
        # Node embedding (from initial positions)
        self.node_embed = MLP([3, hidden_dim, hidden_dim])
        
        # Distance embedding
        self.distance_embed = GaussianSmearing(0.0, cutoff, num_gaussians)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            MLP([time_embed_dim, hidden_dim, hidden_dim]),
        )
        
        # Condition embedding
        self.cond_min = cond_min_value
        self.cond_max = cond_max_value
        self.cond_embed = nn.Sequential(
            SinusoidalTimeEmbedding(cond_embed_dim),
            MLP([cond_embed_dim, hidden_dim, hidden_dim]),
        )
        
        # EGNN layers
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, num_gaussians, update_pos=update_pos)
            for _ in range(num_layers)
        ])
        
        # Output: velocity prediction
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
    
    def forward(
        self, 
        data: Data, 
        t: Tensor, 
        y: Optional[Tensor] = None
    ) -> Tensor:
        pos = data.pos
        edge_index = data.edge_index
        batch = getattr(data, 'batch', torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device))
        
        n_graphs = batch.max().item() + 1 if batch.numel() > 0 else 1
        
        # Node embedding
        h = self.node_embed(pos)
        
        # Time embedding
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(n_graphs)
        time_feat = self.time_embed(t)  # (B, hidden)
        h = h + time_feat[batch]
        
        # Condition embedding
        if y is not None:
            if y.dim() == 2:
                y = y.squeeze(-1)
            # Normalize condition
            y_norm = (y - self.cond_min) / (self.cond_max - self.cond_min)
            cond_feat = self.cond_embed(y_norm)  # (B, hidden)
            h = h + cond_feat[batch]
        
        # EGNN layers
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index, self.distance_embed)
        
        # Output velocity
        velocity = self.output_net(h)
        
        return velocity


class EGNNLayer(nn.Module):
    """Single EGNN layer."""
    
    def __init__(
        self,
        hidden_dim: int,
        n_gaussians: int,
        update_pos: bool = False,
    ):
        super().__init__()
        self.update_pos = update_pos
        
        # Edge MLP
        self.edge_mlp = MLP(
            [hidden_dim * 2 + n_gaussians, hidden_dim, hidden_dim],
        )
        
        # Node MLP
        self.node_mlp = MLP(
            [hidden_dim * 2, hidden_dim, hidden_dim],
            residual=True,
        )
        
        # Coordinate MLP
        if update_pos:
            self.coord_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        h: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        distance_embed: nn.Module,
    ) -> Tuple[Tensor, Tensor]:
        src, dst = edge_index
        
        # Edge vectors and distances
        edge_vec = pos[dst] - pos[src]
        dist = torch.norm(edge_vec, dim=-1, keepdim=True)
        dist_feat = distance_embed(dist.squeeze(-1))
        
        # Edge messages
        edge_input = torch.cat([h[src], h[dst], dist_feat], dim=-1)
        edge_feat = self.edge_mlp(edge_input)
        
        # Aggregate
        agg = scatter(edge_feat, dst, dim=0, dim_size=h.shape[0], reduce='sum')
        
        # Update nodes
        h_new = self.node_mlp(torch.cat([h, agg], dim=-1))
        h_new = self.norm(h_new)
        
        # Update coordinates
        if self.update_pos:
            coord_weights = self.coord_mlp(edge_feat)
            edge_vec_norm = edge_vec / (dist + 1e-8)
            pos_update = scatter(
                coord_weights * edge_vec_norm,
                dst, dim=0, dim_size=pos.shape[0], reduce='sum'
            )
            pos = pos + pos_update
        
        return h_new, pos


# ============================================================================
# SchNet Model
# ============================================================================

class SchNetFlowMatching(FlowMatchingModel):
    """
    SchNet-style model for Flow Matching.
    
    Uses continuous-filter convolutions, good for smooth energy surfaces.
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_filters: int = 64,
        num_gaussians: int = 50,
        cutoff: float = 5.0,
        time_embed_dim: int = 64,
        cond_embed_dim: int = 64,
        cond_min_value: float = 1.0,
        cond_max_value: float = 4.0,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Atom embedding (just positions for amorphous carbon)
        self.atom_embed = MLP([3, hidden_dim, hidden_dim])
        
        # Distance embedding
        self.distance_embed = GaussianSmearing(0.0, cutoff, num_gaussians)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            MLP([time_embed_dim, hidden_dim, hidden_dim]),
        )
        
        # Condition embedding
        self.cond_min = cond_min_value
        self.cond_max = cond_max_value
        self.cond_embed = nn.Sequential(
            SinusoidalTimeEmbedding(cond_embed_dim),
            MLP([cond_embed_dim, hidden_dim, hidden_dim]),
        )
        
        # SchNet interaction blocks
        self.interactions = nn.ModuleList([
            SchNetInteraction(hidden_dim, num_filters, num_gaussians)
            for _ in range(num_layers)
        ])
        
        # Output
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
    
    def forward(
        self, 
        data: Data, 
        t: Tensor, 
        y: Optional[Tensor] = None
    ) -> Tensor:
        pos = data.pos
        edge_index = data.edge_index
        batch = getattr(data, 'batch', torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device))
        
        n_graphs = batch.max().item() + 1 if batch.numel() > 0 else 1
        
        # Node embedding
        h = self.atom_embed(pos)
        
        # Time embedding
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(n_graphs)
        time_feat = self.time_embed(t)
        h = h + time_feat[batch]
        
        # Condition embedding
        if y is not None:
            if y.dim() == 2:
                y = y.squeeze(-1)
            y_norm = (y - self.cond_min) / (self.cond_max - self.cond_min)
            cond_feat = self.cond_embed(y_norm)
            h = h + cond_feat[batch]
        
        # Compute distances
        src, dst = edge_index
        dist = torch.norm(pos[dst] - pos[src], dim=-1)
        dist_feat = self.distance_embed(dist)
        
        # Interaction blocks
        for interaction in self.interactions:
            h = interaction(h, edge_index, dist_feat)
        
        # Output velocity
        velocity = self.output_net(h)
        
        return velocity


class SchNetInteraction(nn.Module):
    """SchNet interaction block."""
    
    def __init__(self, hidden_dim: int, num_filters: int, num_gaussians: int):
        super().__init__()
        
        self.filter_net = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            nn.SiLU(),
            nn.Linear(num_filters, num_filters),
        )
        
        self.atom_net = nn.Sequential(
            nn.Linear(hidden_dim, num_filters),
            nn.SiLU(),
            nn.Linear(num_filters, num_filters),  # Output num_filters to match filter
        )
        
        self.output_net = nn.Sequential(
            nn.Linear(num_filters, hidden_dim),  # Input from aggregated messages
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(
        self,
        h: Tensor,
        edge_index: Tensor,
        dist_feat: Tensor,
    ) -> Tensor:
        src, dst = edge_index
        
        # Continuous filter
        W = self.filter_net(dist_feat)
        
        # Atom-wise transform
        x = self.atom_net(h)
        
        # Filter and aggregate
        msg = W * x[src]
        agg = scatter(msg, dst, dim=0, dim_size=h.shape[0], reduce='sum')
        
        # Output with residual
        return h + self.output_net(agg)


# ============================================================================
# Model Factory
# ============================================================================

# Default configurations for each model type
DEFAULT_CONFIGS = {
    ModelType.NEQUIP: {
        'num_species': 1,
        'cutoff': 5.0,
        'node_dim': 8,
        'edge_num_basis': 16,
        'irreps_hidden': '64x0e + 32x1e + 32x2e',
        'num_convs': 3,
        'radial_neurons': [16, 64],
        'num_neighbors': 12,
        'time_embed_dim': 32,
        'cond_embed_dim': 32,
        'cond_min_value': 1.0,
        'cond_max_value': 4.0,
    },
    ModelType.EGNN: {
        'hidden_dim': 128,
        'num_layers': 4,
        'num_gaussians': 50,
        'cutoff': 5.0,
        'time_embed_dim': 64,
        'cond_embed_dim': 64,
        'cond_min_value': 1.0,
        'cond_max_value': 4.0,
        'update_pos': False,
    },
    ModelType.SCHNET: {
        'hidden_dim': 128,
        'num_layers': 4,
        'num_filters': 64,
        'num_gaussians': 50,
        'cutoff': 5.0,
        'time_embed_dim': 64,
        'cond_embed_dim': 64,
        'cond_min_value': 1.0,
        'cond_max_value': 4.0,
    },
}


def create_model(
    model_type: Union[str, ModelType],
    **kwargs,
) -> FlowMatchingModel:
    """
    Create a flow matching model.
    
    Args:
        model_type: Type of model ('nequip', 'egnn', 'schnet')
        **kwargs: Model-specific configuration
        
    Returns:
        Initialized model
        
    Example:
        >>> model = create_model('nequip', num_convs=4)
        >>> model = create_model('egnn', hidden_dim=256)
    """
    if isinstance(model_type, str):
        model_type = ModelType.from_string(model_type)
    
    # Get default config and merge with kwargs
    config = DEFAULT_CONFIGS[model_type].copy()
    config.update(kwargs)
    
    if model_type == ModelType.NEQUIP:
        # Import here to avoid circular imports
        from diffcsp.pl_modules.nequip_flow import create_nequip_flow_model
        return create_nequip_flow_model(**config)
    
    elif model_type == ModelType.EGNN:
        return EGNNFlowMatching(**config)
    
    elif model_type == ModelType.SCHNET:
        return SchNetFlowMatching(**config)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_model_info(model_type: Union[str, ModelType]) -> Dict[str, Any]:
    """Get information about a model type."""
    if isinstance(model_type, str):
        model_type = ModelType.from_string(model_type)
    
    info = {
        ModelType.NEQUIP: {
            'name': 'NequIP',
            'description': 'E(3) equivariant neural network using spherical harmonics',
            'pros': ['Highest accuracy', 'Full rotation equivariance', 'Strong for geometry'],
            'cons': ['Slowest', 'Requires e3nn package'],
            'params_estimate': '200k-1M',
        },
        ModelType.EGNN: {
            'name': 'EGNN',
            'description': 'E(n) equivariant graph neural network',
            'pros': ['Fast', 'Good accuracy', 'No special packages needed'],
            'cons': ['Less expressive than NequIP'],
            'params_estimate': '100k-500k',
        },
        ModelType.SCHNET: {
            'name': 'SchNet',
            'description': 'Continuous-filter convolutional network',
            'pros': ['Very fast', 'Simple architecture', 'Good for smooth potentials'],
            'cons': ['Not equivariant', 'May need data augmentation'],
            'params_estimate': '50k-300k',
        },
    }
    
    return info[model_type]


def list_available_models() -> List[str]:
    """List all available model types."""
    return [mt.value for mt in ModelType]


# ============================================================================
# Testing
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Model Factory Test")
    print("=" * 60)
    
    # Create dummy data
    from torch_geometric.data import Data, Batch
    
    num_atoms = 50
    pos = torch.rand(num_atoms, 3) * 12.0
    pos[:, 2] = 0  # 2D
    
    edge_index = torch.randint(0, num_atoms, (2, 200))
    edge_attr = torch.randn(200, 3)
    
    data = Data(
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=torch.zeros(num_atoms, dtype=torch.long),
    )
    
    t = torch.tensor([0.5])
    y = torch.tensor([2.0])  # log10(100 K/ps)
    
    print("\nTesting models:")
    print("-" * 40)
    
    for model_name in list_available_models():
        print(f"\n{model_name.upper()}:")
        
        try:
            model = create_model(model_name)
            
            # Count parameters
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {n_params:,}")
            
            # Forward pass
            with torch.no_grad():
                v = model(data, t, y)
            
            print(f"  Output shape: {v.shape}")
            print(f"  Output range: [{v.min():.4f}, {v.max():.4f}]")
            
            # Get info
            info = get_model_info(model_name)
            print(f"  Description: {info['description']}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
