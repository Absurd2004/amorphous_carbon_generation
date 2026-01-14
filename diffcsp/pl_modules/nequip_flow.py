"""
NequIP-based Flow Matching Model for 2D Amorphous Carbon Generation.

This module adapts the E3-equivariant NequIP architecture from DM2 for Flow Matching,
replacing the denoising diffusion approach with optimal transport flow matching.

Key differences from DM2:
1. Denoising Diffusion → Flow Matching (OT-CFM)
2. Predicts velocity field v(x_t, t) instead of noise
3. Supports conditional generation (cooling rate)

Author: Adapted from DM2/src/graphite/nn/models/e3nn_nequip.py
"""

import torch
import torch.nn as nn
from functools import partial

from e3nn import o3
from e3nn.nn import Gate

try:
    from graphite.nn.conv.e3nn_nequip import Interaction
    from graphite.nn.basis import bessel
except ImportError:
    # Fallback: define locally if DM2 not in path
    import sys
    sys.path.insert(0, '/home/yongkunyang/DM2/src')
    from graphite.nn.conv.e3nn_nequip import Interaction
    from graphite.nn.basis import bessel


def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    """Check if a tensor product path exists between two irreps to produce a given output irrep."""
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)
    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


class Compose(nn.Module):
    """Compose two modules sequentially."""
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second
        self.irreps_in = self.first.irreps_in
        self.irreps_out = self.second.irreps_out

    def forward(self, *input):
        x = self.first(*input)
        return self.second(x)


class GaussianBasisEmbedding(nn.Module):
    """
    Embeds a scalar value using a Gaussian basis set followed by a dense layer.
    
    Used for embedding:
    - Time t ∈ [0, 1]
    - Cooling rate (log10 scale) ∈ [-4, 4]
    
    Args:
        num_basis: Number of Gaussian basis functions
        embedding_dim: Dimension of the final embedding vector
        min_sigma: Minimum sigma (width) for the Gaussian bases
        min_value: Minimum value of the input range
        max_value: Maximum value of the input range
    """
    def __init__(
        self, 
        num_basis=12, 
        embedding_dim=32, 
        min_sigma=0.1, 
        learn_means=False, 
        learn_sigmas=False,
        min_value=0,
        max_value=1,
    ):
        super().__init__()
        
        # Initialize means uniformly in [min_value, max_value]
        means = torch.linspace(min_value, max_value, num_basis)
        if learn_means:
            self.means = nn.Parameter(means)
        else:
            self.register_buffer('means', means)
            
        # Initialize sigmas with reasonable defaults
        sigmas = torch.ones_like(means) * max(min_sigma, 1.0 / (num_basis - 1))
        if learn_sigmas:
            self.sigmas = nn.Parameter(sigmas)
        else:
            self.register_buffer('sigmas', sigmas)
            
        # Two-layer neural network to produce the final embedding
        hidden_dim = max(embedding_dim * 2, num_basis)
        self.layer1 = nn.Linear(num_basis, hidden_dim)
        self.activation = nn.SiLU()
        self.layer2 = nn.Linear(hidden_dim, embedding_dim)
        
    def gaussian_basis(self, x):
        """Transform scalar input into Gaussian basis activations."""
        if x.dim() == 1:
            x = x.unsqueeze(1)
        x_expanded = x.expand(-1, self.means.shape[0])
        return torch.exp(-0.5 * ((x_expanded - self.means) / self.sigmas)**2)
        
    def forward(self, x):
        """
        Forward pass of the embedding module.
        
        Args:
            x: Input tensor of shape [batch_size] or [batch_size, 1]
               
        Returns:
            Embedded representation of shape [batch_size, embedding_dim]
        """
        basis_activation = self.gaussian_basis(x)
        hidden = self.activation(self.layer1(basis_activation))
        embedding = self.layer2(hidden)
        return embedding


class InitialEmbedding(nn.Module):
    """
    Initial embedding layer for the NequIP network.
    
    Embeds:
    - Node features (atom types) → h_node_x, h_node_z
    - Edge features (bond distances) → h_edge
    """
    def __init__(self, num_species, cutoff, node_dim=8, edge_num_basis=16):
        super().__init__()
        self.embed_node_x = nn.Embedding(num_species, node_dim)
        self.embed_node_z = nn.Embedding(num_species, node_dim)
        self.embed_edge = partial(bessel, start=0.0, end=cutoff, num_basis=edge_num_basis)
    
    def forward(self, data):
        data.h_node_x = self.embed_node_x(data.x)
        data.h_node_z = self.embed_node_z(data.x)
        data.h_edge = self.embed_edge(data.edge_attr.norm(dim=-1))
        return data


class NequIP_FlowMatching(nn.Module):
    """
    NequIP-based Flow Matching Model for Amorphous Carbon Generation.
    
    This model combines:
    1. E3-equivariant message passing (NequIP architecture)
    2. Time embedding for flow matching
    3. Condition embedding (cooling rate)
    
    The model predicts velocity field v(x_t, t, y) where:
    - x_t: Interpolated atomic positions at time t
    - t: Flow time ∈ [0, 1]
    - y: Condition (e.g., log10 cooling rate)
    
    Architecture:
        Input → Initial Embedding → [Interaction + Gate + Time/Cond Injection] × N → Output
    
    Args:
        init_embed: Initial embedding function/class
        irreps_node_x: Irreps of input node features (default: '8x0e')
        irreps_node_z: Irreps of auxiliary node features (default: '8x0e')
        irreps_hidden: Irreps of hidden node features (default: '64x0e + 32x1e + 32x2e')
        irreps_edge: Irreps of edge spherical harmonics (default: '1x0e + 1x1e + 1x2e')
        irreps_out: Irreps of output (default: '1x1e' for 3D velocity)
        num_convs: Number of interaction layers (default: 3)
        radial_neurons: MLP architecture for radial basis (default: [16, 64])
        num_neighbors: Average number of neighbors for normalization (default: 12)
        time_embed_dim: Dimension of time embedding (default: 32)
        cond_embed_dim: Dimension of condition embedding (default: 32)
        cond_min_value: Min value for condition (log10 cooling rate) (default: -1)
        cond_max_value: Max value for condition (log10 cooling rate) (default: 3)
    """
    def __init__(
        self,
        init_embed,
        irreps_node_x='8x0e',
        irreps_node_z='8x0e',
        irreps_hidden='64x0e + 32x1e + 32x2e',
        irreps_edge='1x0e + 1x1e + 1x2e',
        irreps_out='1x1e',
        num_convs=3,
        radial_neurons=[16, 64],
        num_neighbors=12,
        # Flow Matching specific
        time_embed_dim=32,
        time_num_basis=12,
        # Condition specific (cooling rate)
        cond_embed_dim=32,
        cond_num_basis=9,
        cond_min_value=-1.0,  # log10(0.1 K/ps)
        cond_max_value=3.0,   # log10(1000 K/ps)
    ):
        super().__init__()
        
        self.init_embed = init_embed
        self.irreps_node_x = o3.Irreps(irreps_node_x)
        self.irreps_node_z = o3.Irreps(irreps_node_z)
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_edge = o3.Irreps(irreps_edge)
        self.num_convs = num_convs
        
        act_scalars = {1: nn.functional.silu, -1: torch.tanh}
        act_gates = {1: torch.sigmoid, -1: torch.tanh}
        
        # Build interaction layers
        irreps = self.irreps_node_x
        self.interactions = nn.ModuleList()
        
        for _ in range(num_convs):
            irreps_scalars = o3.Irreps([
                (m, ir) for m, ir in self.irreps_hidden 
                if ir.l == 0 and tp_path_exists(irreps, self.irreps_edge, ir)
            ])
            irreps_gated = o3.Irreps([
                (m, ir) for m, ir in self.irreps_hidden 
                if ir.l > 0 and tp_path_exists(irreps, self.irreps_edge, ir)
            ])
            
            if irreps_gated.dim > 0:
                if tp_path_exists(irreps_node_z, self.irreps_edge, "0e"):
                    ir = "0e"
                elif tp_path_exists(irreps_node_z, self.irreps_edge, "0o"):
                    ir = "0o"
                else:
                    raise ValueError(
                        f"irreps={irreps} times irreps_edge={self.irreps_edge} "
                        f"is unable to produce gates for irreps_gated={irreps_gated}."
                    )
            else:
                ir = None
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()
            
            gate = Gate(
                irreps_scalars, [act_scalars[ir.p] for _, ir in irreps_scalars],
                irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],
                irreps_gated
            )
            
            conv = Interaction(
                irreps_in=irreps,
                irreps_node=self.irreps_node_z,
                irreps_edge=self.irreps_edge,
                irreps_out=gate.irreps_in,
                radial_neurons=radial_neurons,
                num_neighbors=num_neighbors,
            )
            irreps = gate.irreps_out
            self.interactions.append(Compose(conv, gate))
        
        # Final output layer (produces 3D velocity vector)
        self.out = o3.FullyConnectedTensorProduct(
            irreps_in1=irreps,
            irreps_in2=self.irreps_node_z,
            irreps_out=self.irreps_out,
        )
        
        # Get the hidden dimension for embeddings
        hidden_dim = irreps.dim
        
        # Time embedding for Flow Matching (t ∈ [0, 1])
        self.t_embed = GaussianBasisEmbedding(
            num_basis=time_num_basis,
            embedding_dim=time_embed_dim,
            min_value=0.0,
            max_value=1.0,
            min_sigma=0.1,
        )
        self.t_projection = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Condition embedding (cooling rate, log10 scale)
        self.y_embed = GaussianBasisEmbedding(
            num_basis=cond_num_basis,
            embedding_dim=cond_embed_dim,
            min_value=cond_min_value,
            max_value=cond_max_value,
            min_sigma=0.6,
        )
        self.y_projection = nn.Sequential(
            nn.Linear(cond_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
    def forward(self, data, t, y=None):
        """
        Forward pass of the NequIP Flow Matching model.
        
        Args:
            data: PyG Data object with:
                - x: Atom types [num_atoms]
                - pos: Atom positions at time t [num_atoms, 3]
                - edge_index: Graph connectivity [2, num_edges]
                - edge_attr: Edge vectors [num_edges, 3]
                - batch: Batch assignment [num_atoms]
            t: Flow time tensor [batch_size] or scalar
            y: Condition (log10 cooling rate) [batch_size], optional
            
        Returns:
            velocity: Predicted velocity field [num_atoms, 3]
        """
        # Initial embedding
        data = self.init_embed(data)
        edge_index, edge_attr = data.edge_index, data.edge_attr
        h_node_x, h_node_z, h_edge = data.h_node_x, data.h_node_z, data.h_edge
        
        # Get batch assignment
        batch = getattr(data, 'batch', torch.zeros(h_node_x.size(0), device=h_node_x.device, dtype=torch.long))
        num_graphs = batch.max().item() + 1 if batch.numel() > 0 else 1
        
        # Time embedding
        if isinstance(t, (int, float)):
            t = torch.tensor([t], device=h_node_x.device).expand(num_graphs)
        elif t.dim() == 0:
            t = t.unsqueeze(0).expand(num_graphs)
        
        h_t = self.t_embed(t)  # [batch_size, embed_dim]
        h_t = h_t[batch]  # [num_atoms, embed_dim]
        h_t = self.t_projection(h_t)  # [num_atoms, hidden_dim]
        
        # Condition embedding (cooling rate)
        if y is not None:
            if isinstance(y, (int, float)):
                y = torch.tensor([y], device=h_node_x.device).expand(num_graphs)
            elif y.dim() == 0:
                y = y.unsqueeze(0).expand(num_graphs)
            h_y = self.y_embed(y)  # [batch_size, embed_dim]
            h_y = h_y[batch]  # [num_atoms, embed_dim]
            h_y = self.y_projection(h_y)  # [num_atoms, hidden_dim]
        else:
            h_y = 0
        
        # Edge spherical harmonics
        edge_sh = o3.spherical_harmonics(
            self.irreps_edge, edge_attr, 
            normalize=True, normalization='component'
        )
        
        # Graph convolutions with time and condition injection
        for layer in self.interactions:
            h_node_x = layer(h_node_x, h_node_z, edge_index, edge_sh, h_edge)
            h_node_x = h_node_x + h_t + h_y  # Additive injection
        
        # Final output: velocity field
        velocity = self.out(h_node_x, h_node_z)
        
        return velocity
    
    def predict_velocity(self, data, t, y=None):
        """Alias for forward() for clarity in flow matching context."""
        return self.forward(data, t, y)


def create_nequip_flow_model(
    num_species=1,  # Only carbon
    cutoff=5.0,
    node_dim=8,
    edge_num_basis=16,
    irreps_hidden='64x0e + 32x1e + 32x2e',
    num_convs=3,
    radial_neurons=[16, 64],
    num_neighbors=12,
    time_embed_dim=32,
    cond_embed_dim=32,
    cond_min_value=-1.0,
    cond_max_value=3.0,
):
    """
    Factory function to create a NequIP Flow Matching model.
    
    Args:
        num_species: Number of atom types (1 for pure carbon)
        cutoff: Cutoff radius for graph construction
        node_dim: Dimension of node embeddings
        edge_num_basis: Number of Bessel basis functions for edge embedding
        irreps_hidden: Hidden layer irreps
        num_convs: Number of interaction layers
        radial_neurons: MLP architecture for radial basis
        num_neighbors: Average number of neighbors
        time_embed_dim: Time embedding dimension
        cond_embed_dim: Condition embedding dimension
        cond_min_value: Min log10 cooling rate
        cond_max_value: Max log10 cooling rate
        
    Returns:
        NequIP_FlowMatching model
    """
    init_embed = InitialEmbedding(
        num_species=num_species,
        cutoff=cutoff,
        node_dim=node_dim,
        edge_num_basis=edge_num_basis,
    )
    
    model = NequIP_FlowMatching(
        init_embed=init_embed,
        irreps_node_x=f'{node_dim}x0e',
        irreps_node_z=f'{node_dim}x0e',
        irreps_hidden=irreps_hidden,
        irreps_edge='1x0e + 1x1e + 1x2e',
        irreps_out='1x1e',  # 3D velocity vector
        num_convs=num_convs,
        radial_neurons=radial_neurons,
        num_neighbors=num_neighbors,
        time_embed_dim=time_embed_dim,
        cond_embed_dim=cond_embed_dim,
        cond_min_value=cond_min_value,
        cond_max_value=cond_max_value,
    )
    
    return model


if __name__ == "__main__":
    # Quick test
    print("Testing NequIP_FlowMatching...")
    
    model = create_nequip_flow_model(
        num_species=1,
        cutoff=5.0,
        num_convs=2,
    )
    print(f"Model created successfully!")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
