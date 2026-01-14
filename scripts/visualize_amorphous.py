#!/usr/bin/env python
"""
Visualization Tools for 2D Amorphous Carbon Structures.

Provides:
1. Structure visualization (2D/3D plots)
2. Trajectory animation (ODE sampling)
3. Comparison of generated vs reference
4. Batch visualization

Usage:
    # Visualize a single structure
    python scripts/visualize_amorphous.py --input sample.data --output plot.png
    
    # Visualize multiple structures in a grid
    python scripts/visualize_amorphous.py --input generated/ --output grid.png --grid
    
    # Compare generated vs reference
    python scripts/visualize_amorphous.py --generated gen.data --reference ref.data --output compare.png
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def read_lammps_positions(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read positions and box from LAMMPS data file.
    
    Returns:
        positions: [num_atoms, 3]
        box: [3] (box dimensions)
    """
    positions = []
    box = [12.0, 12.0, 20.0]
    
    with open(filepath, 'r') as f:
        in_atoms = False
        for line in f:
            line = line.strip()
            
            if 'xlo xhi' in line:
                parts = line.split()
                box[0] = float(parts[1]) - float(parts[0])
            elif 'ylo yhi' in line:
                parts = line.split()
                box[1] = float(parts[1]) - float(parts[0])
            elif 'zlo zhi' in line:
                parts = line.split()
                box[2] = float(parts[1]) - float(parts[0])
            elif line.startswith('Atoms'):
                in_atoms = True
                continue
            elif in_atoms and line:
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                        positions.append([x, y, z])
                    except (ValueError, IndexError):
                        continue
    
    return np.array(positions), np.array(box)


def minimum_image_distance(pos1: np.ndarray, pos2: np.ndarray, box: np.ndarray) -> float:
    """Compute minimum image distance under PBC."""
    diff = pos1 - pos2
    diff = diff - box * np.round(diff / box)
    return np.linalg.norm(diff)


def compute_bonds(
    positions: np.ndarray,
    box: np.ndarray,
    cutoff: float = 1.85,
) -> List[Tuple[int, int]]:
    """Find bonds between atoms."""
    n_atoms = len(positions)
    bonds = []
    
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = minimum_image_distance(positions[i], positions[j], box)
            if dist < cutoff:
                bonds.append((i, j))
    
    return bonds


def get_coordination_colors(positions: np.ndarray, box: np.ndarray, cutoff: float = 1.85) -> np.ndarray:
    """Get colors based on coordination number."""
    n_atoms = len(positions)
    coord = np.zeros(n_atoms)
    
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i != j:
                dist = minimum_image_distance(positions[i], positions[j], box)
                if dist < cutoff:
                    coord[i] += 1
    
    # Color mapping: 2=blue (sp), 3=green (sp2), 4=red (sp3), other=gray
    colors = []
    color_map = {
        1: 'purple',
        2: 'blue',     # sp
        3: 'green',    # sp2
        4: 'red',      # sp3
    }
    
    for c in coord:
        colors.append(color_map.get(int(c), 'gray'))
    
    return colors, coord


def plot_structure_2d(
    positions: np.ndarray,
    box: np.ndarray,
    ax: Optional[plt.Axes] = None,
    show_bonds: bool = True,
    show_box: bool = True,
    color_by_coord: bool = True,
    title: str = "",
    atom_size: float = 100,
    bond_cutoff: float = 1.85,
) -> plt.Axes:
    """
    Plot 2D projection of structure.
    
    Args:
        positions: Atomic positions [N, 3]
        box: Box dimensions [3]
        ax: Matplotlib axes (created if None)
        show_bonds: Whether to draw bonds
        show_box: Whether to draw box outline
        color_by_coord: Color atoms by coordination number
        title: Plot title
        atom_size: Size of atom markers
        bond_cutoff: Cutoff for bonds
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    x = positions[:, 0]
    y = positions[:, 1]
    
    # Get colors
    if color_by_coord:
        colors, coord = get_coordination_colors(positions, box, bond_cutoff)
    else:
        colors = 'black'
    
    # Draw bonds
    if show_bonds:
        bonds = compute_bonds(positions, box, bond_cutoff)
        segments = []
        for i, j in bonds:
            # Handle PBC crossing
            diff = positions[j][:2] - positions[i][:2]
            diff = diff - box[:2] * np.round(diff / box[:2])
            
            # Only draw if not crossing boundary
            if np.all(np.abs(diff) < box[:2] / 2):
                segments.append([positions[i][:2], positions[j][:2]])
        
        lc = LineCollection(segments, colors='gray', linewidths=1, alpha=0.5, zorder=1)
        ax.add_collection(lc)
    
    # Draw atoms
    ax.scatter(x, y, c=colors, s=atom_size, edgecolors='black', linewidths=0.5, zorder=2)
    
    # Draw box
    if show_box:
        rect = plt.Rectangle((0, 0), box[0], box[1], fill=False, 
                              edgecolor='black', linewidth=2, zorder=0)
        ax.add_patch(rect)
    
    # Set limits
    padding = 0.5
    ax.set_xlim(-padding, box[0] + padding)
    ax.set_ylim(-padding, box[1] + padding)
    ax.set_aspect('equal')
    
    ax.set_xlabel('x (Å)', fontsize=12)
    ax.set_ylabel('y (Å)', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14)
    
    return ax


def plot_structure_grid(
    positions_list: List[np.ndarray],
    box: np.ndarray,
    titles: Optional[List[str]] = None,
    ncols: int = 4,
    figsize: Optional[Tuple[int, int]] = None,
    output_path: Optional[str] = None,
    **kwargs,
):
    """
    Plot multiple structures in a grid.
    
    Args:
        positions_list: List of position arrays
        box: Box dimensions
        titles: Optional list of titles
        ncols: Number of columns
        figsize: Figure size
        output_path: Path to save figure
    """
    n = len(positions_list)
    nrows = (n + ncols - 1) // ncols
    
    if figsize is None:
        figsize = (4 * ncols, 4 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)
    
    for idx, pos in enumerate(positions_list):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]
        
        title = titles[idx] if titles else f"Sample {idx + 1}"
        plot_structure_2d(pos, box, ax=ax, title=title, **kwargs)
    
    # Hide empty axes
    for idx in range(n, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Grid plot saved to: {output_path}")
    
    return fig


def plot_comparison(
    gen_positions: np.ndarray,
    ref_positions: np.ndarray,
    box: np.ndarray,
    output_path: Optional[str] = None,
    gen_title: str = "Generated",
    ref_title: str = "Reference",
):
    """
    Side-by-side comparison of generated and reference structures.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    plot_structure_2d(gen_positions, box, ax=axes[0], title=gen_title)
    plot_structure_2d(ref_positions, box, ax=axes[1], title=ref_title)
    
    # Add legend for coordination colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label='sp (2 bonds)'),
        Patch(facecolor='green', label='sp² (3 bonds)'),
        Patch(facecolor='red', label='sp³ (4 bonds)'),
        Patch(facecolor='gray', label='Other'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4, 
               bbox_to_anchor=(0.5, 0.02), fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_path}")
    
    return fig


def plot_trajectory(
    trajectory: List[np.ndarray],
    box: np.ndarray,
    output_path: Optional[str] = None,
    num_frames: int = 9,
    figsize: Tuple[int, int] = (15, 15),
):
    """
    Visualize ODE sampling trajectory.
    
    Args:
        trajectory: List of position arrays at each time step
        box: Box dimensions
        output_path: Path to save figure
        num_frames: Number of frames to show
    """
    n_steps = len(trajectory)
    indices = np.linspace(0, n_steps - 1, num_frames, dtype=int)
    
    ncols = 3
    nrows = (num_frames + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        t = idx / (n_steps - 1)
        plot_structure_2d(
            trajectory[idx], box, ax=axes[i],
            title=f"t = {t:.2f}",
            color_by_coord=True,
        )
    
    # Hide unused axes
    for i in range(num_frames, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle("ODE Sampling Trajectory (t: 0 → 1)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Trajectory plot saved to: {output_path}")
    
    return fig


def plot_cooling_rate_comparison(
    samples_by_rate: Dict[int, List[np.ndarray]],
    box: np.ndarray,
    output_path: Optional[str] = None,
):
    """
    Compare structures across different cooling rates.
    
    Args:
        samples_by_rate: Dict mapping cooling rate to list of position arrays
        box: Box dimensions
        output_path: Path to save figure
    """
    rates = sorted(samples_by_rate.keys())
    n_rates = len(rates)
    
    # Show one sample per rate
    fig, axes = plt.subplots(1, n_rates, figsize=(4 * n_rates, 4))
    if n_rates == 1:
        axes = [axes]
    
    for i, rate in enumerate(rates):
        samples = samples_by_rate[rate]
        if len(samples) > 0:
            plot_structure_2d(
                samples[0], box, ax=axes[i],
                title=f"{rate} K/ps",
            )
    
    plt.suptitle("Cooling Rate Comparison", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Cooling rate comparison saved to: {output_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Visualize amorphous carbon structures"
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input file or directory'
    )
    parser.add_argument(
        '--generated', '-g',
        type=str,
        help='Generated sample file (for comparison)'
    )
    parser.add_argument(
        '--reference', '-r',
        type=str,
        help='Reference sample file (for comparison)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='visualization.png',
        help='Output file path'
    )
    parser.add_argument(
        '--grid',
        action='store_true',
        help='Create grid visualization'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=16,
        help='Maximum samples for grid'
    )
    parser.add_argument(
        '--no_bonds',
        action='store_true',
        help='Do not show bonds'
    )
    parser.add_argument(
        '--no_color',
        action='store_true',
        help='Do not color by coordination'
    )
    
    args = parser.parse_args()
    
    # Mode 1: Comparison
    if args.generated and args.reference:
        gen_pos, box = read_lammps_positions(args.generated)
        ref_pos, _ = read_lammps_positions(args.reference)
        
        plot_comparison(gen_pos, ref_pos, box, args.output)
        return
    
    # Mode 2: Single or grid
    if args.input:
        input_path = Path(args.input)
        
        if input_path.is_file():
            # Single file
            positions, box = read_lammps_positions(str(input_path))
            
            fig, ax = plt.subplots(figsize=(8, 8))
            plot_structure_2d(
                positions, box, ax=ax,
                title=input_path.name,
                show_bonds=not args.no_bonds,
                color_by_coord=not args.no_color,
            )
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='blue', label='sp (2)'),
                Patch(facecolor='green', label='sp² (3)'),
                Patch(facecolor='red', label='sp³ (4)'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
            
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"Structure plot saved to: {args.output}")
            
        elif input_path.is_dir():
            # Directory - grid visualization
            files = list(input_path.glob("**/*.data"))[:args.max_samples]
            
            if len(files) == 0:
                print(f"No .data files found in {input_path}")
                return
            
            positions_list = []
            titles = []
            box = None
            
            for f in files:
                pos, b = read_lammps_positions(str(f))
                if len(pos) > 0:
                    positions_list.append(pos)
                    titles.append(f.name)
                    if box is None:
                        box = b
            
            if len(positions_list) > 0:
                plot_structure_grid(
                    positions_list, box,
                    titles=titles,
                    output_path=args.output,
                    show_bonds=not args.no_bonds,
                    color_by_coord=not args.no_color,
                )
        else:
            print(f"Input not found: {args.input}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
