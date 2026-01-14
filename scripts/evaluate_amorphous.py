#!/usr/bin/env python
"""
Evaluation Metrics for 2D Amorphous Carbon Structures.

This module provides comprehensive metrics for evaluating generated
amorphous carbon structures:
1. Radial Distribution Function (RDF)
2. Bond Angle Distribution
3. Coordination Number Distribution
4. Ring Statistics (3-8 member rings)
5. Structure Comparison (generated vs reference)

Usage:
    python scripts/evaluate_amorphous.py \
        --generated generated/samples/ \
        --reference data/amorphous_carbon/data/ \
        --output evaluation_results/
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
from collections import Counter, defaultdict
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from ase import Atoms
    from ase.io import read
    from ase.neighborlist import natural_cutoffs, NeighborList
    HAS_ASE = True
except ImportError:
    HAS_ASE = False
    print("Warning: ASE not available. Some features may be limited.")

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: NetworkX not available. Ring statistics will be disabled.")


# Physical constants
CARBON_BOND_LENGTH = 1.42  # Graphene C-C bond length in Å
CARBON_BOND_CUTOFF = 1.85  # Cutoff for considering a C-C bond


def read_lammps_positions(filepath: str) -> np.ndarray:
    """
    Read atomic positions from LAMMPS data file.
    
    Args:
        filepath: Path to LAMMPS data file
        
    Returns:
        positions: Atomic positions [num_atoms, 3]
    """
    positions = []
    in_atoms_section = False
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('Atoms'):
                in_atoms_section = True
                continue
            
            if in_atoms_section and line:
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        # Format: atom_id type x y z
                        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                        positions.append([x, y, z])
                    except (ValueError, IndexError):
                        continue
    
    return np.array(positions)


def read_box_size(filepath: str) -> np.ndarray:
    """Read box size from LAMMPS data file."""
    box = []
    with open(filepath, 'r') as f:
        for line in f:
            if 'xlo xhi' in line:
                parts = line.split()
                box.append(float(parts[1]) - float(parts[0]))
            elif 'ylo yhi' in line:
                parts = line.split()
                box.append(float(parts[1]) - float(parts[0]))
            elif 'zlo zhi' in line:
                parts = line.split()
                box.append(float(parts[1]) - float(parts[0]))
    return np.array(box) if len(box) == 3 else np.array([12.0, 12.0, 20.0])


def minimum_image_distance(pos1: np.ndarray, pos2: np.ndarray, box: np.ndarray) -> float:
    """Compute minimum image distance under PBC."""
    diff = pos1 - pos2
    diff = diff - box * np.round(diff / box)
    return np.linalg.norm(diff)


def compute_pairwise_distances(
    positions: np.ndarray, 
    box: np.ndarray,
    pbc: bool = True
) -> np.ndarray:
    """
    Compute all pairwise distances with periodic boundary conditions.
    
    Args:
        positions: Atomic positions [num_atoms, 3]
        box: Box dimensions [3]
        pbc: Whether to apply periodic boundary conditions
        
    Returns:
        distances: Upper triangular distance matrix entries
    """
    n_atoms = len(positions)
    distances = []
    
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if pbc:
                dist = minimum_image_distance(positions[i], positions[j], box)
            else:
                dist = np.linalg.norm(positions[i] - positions[j])
            distances.append(dist)
    
    return np.array(distances)


def compute_rdf(
    positions: np.ndarray,
    box: np.ndarray,
    r_max: float = 6.0,
    n_bins: int = 100,
    pbc: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Radial Distribution Function.
    
    g(r) = (V / N^2) * <n(r)> / (4πr²dr)
    For 2D: g(r) = (A / N^2) * <n(r)> / (2πr*dr)
    
    Args:
        positions: Atomic positions [num_atoms, 3]
        box: Box dimensions [3]
        r_max: Maximum distance to compute RDF
        n_bins: Number of bins
        pbc: Whether to apply periodic boundary conditions
        
    Returns:
        r: Radial distances
        g_r: RDF values
    """
    n_atoms = len(positions)
    
    # Compute pairwise distances
    distances = compute_pairwise_distances(positions, box, pbc)
    
    # Bin the distances
    dr = r_max / n_bins
    r = np.linspace(dr/2, r_max - dr/2, n_bins)
    hist, bin_edges = np.histogram(distances, bins=n_bins, range=(0, r_max))
    
    # For 2D system (z=0), use 2D normalization
    # Check if 2D (all z coordinates are ~0)
    is_2d = np.std(positions[:, 2]) < 0.1
    
    if is_2d:
        # 2D normalization: shell area = 2πr*dr
        area = box[0] * box[1]
        density_2d = n_atoms / area
        
        # Each bin has area 2πr*dr
        bin_area = 2 * np.pi * r * dr
        expected = bin_area * density_2d * n_atoms / 2  # /2 for double counting
        
        # Avoid division by zero
        g_r = np.zeros_like(r)
        nonzero = expected > 0
        g_r[nonzero] = hist[nonzero] / expected[nonzero]
    else:
        # 3D normalization
        volume = np.prod(box)
        density = n_atoms / volume
        
        # Shell volume = 4πr²dr
        shell_volume = 4 * np.pi * r**2 * dr
        expected = shell_volume * density * n_atoms / 2
        
        g_r = np.zeros_like(r)
        nonzero = expected > 0
        g_r[nonzero] = hist[nonzero] / expected[nonzero]
    
    return r, g_r


def compute_neighbors(
    positions: np.ndarray,
    box: np.ndarray,
    cutoff: float = CARBON_BOND_CUTOFF,
    pbc: bool = True,
) -> List[List[int]]:
    """
    Compute neighbor list for each atom.
    
    Args:
        positions: Atomic positions [num_atoms, 3]
        box: Box dimensions [3]
        cutoff: Bond cutoff distance
        pbc: Whether to apply periodic boundary conditions
        
    Returns:
        neighbors: List of neighbor indices for each atom
    """
    n_atoms = len(positions)
    neighbors = [[] for _ in range(n_atoms)]
    
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if pbc:
                dist = minimum_image_distance(positions[i], positions[j], box)
            else:
                dist = np.linalg.norm(positions[i] - positions[j])
            
            if dist < cutoff:
                neighbors[i].append(j)
                neighbors[j].append(i)
    
    return neighbors


def compute_coordination_numbers(neighbors: List[List[int]]) -> np.ndarray:
    """
    Compute coordination number for each atom.
    
    Args:
        neighbors: Neighbor list from compute_neighbors()
        
    Returns:
        coordination: Coordination number for each atom
    """
    return np.array([len(n) for n in neighbors])


def compute_bond_angles(
    positions: np.ndarray,
    box: np.ndarray,
    neighbors: List[List[int]],
    pbc: bool = True,
) -> np.ndarray:
    """
    Compute all C-C-C bond angles.
    
    For each atom with 2+ neighbors, compute angles between all pairs.
    
    Args:
        positions: Atomic positions [num_atoms, 3]
        box: Box dimensions [3]
        neighbors: Neighbor list
        pbc: Whether to apply periodic boundary conditions
        
    Returns:
        angles: All bond angles in degrees
    """
    angles = []
    
    for i, neigh_i in enumerate(neighbors):
        if len(neigh_i) < 2:
            continue
        
        # Get vectors to all neighbors
        vectors = []
        for j in neigh_i:
            diff = positions[j] - positions[i]
            if pbc:
                diff = diff - box * np.round(diff / box)
            vectors.append(diff / np.linalg.norm(diff))
        
        # Compute angles between all pairs of neighbors
        for k1 in range(len(vectors)):
            for k2 in range(k1 + 1, len(vectors)):
                cos_angle = np.dot(vectors[k1], vectors[k2])
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.degrees(np.arccos(cos_angle))
                angles.append(angle)
    
    return np.array(angles)


def compute_ring_statistics(
    positions: np.ndarray,
    box: np.ndarray,
    neighbors: List[List[int]],
    max_ring_size: int = 8,
) -> Dict[int, int]:
    """
    Compute ring statistics (number of rings of each size).
    
    Uses NetworkX cycle detection.
    
    Args:
        positions: Atomic positions [num_atoms, 3]
        box: Box dimensions [3]
        neighbors: Neighbor list
        max_ring_size: Maximum ring size to count
        
    Returns:
        ring_counts: Dictionary mapping ring size to count
    """
    if not HAS_NETWORKX:
        return {}
    
    # Build graph
    G = nx.Graph()
    n_atoms = len(positions)
    G.add_nodes_from(range(n_atoms))
    
    for i, neigh_i in enumerate(neighbors):
        for j in neigh_i:
            if i < j:
                G.add_edge(i, j)
    
    # Find all cycles up to max_ring_size
    ring_counts = defaultdict(int)
    
    try:
        # Find cycle basis (minimum cycles)
        cycles = nx.cycle_basis(G)
        
        for cycle in cycles:
            size = len(cycle)
            if 3 <= size <= max_ring_size:
                ring_counts[size] += 1
    except Exception:
        pass
    
    return dict(ring_counts)


def compute_all_metrics(
    positions: np.ndarray,
    box: np.ndarray,
    bond_cutoff: float = CARBON_BOND_CUTOFF,
) -> Dict[str, any]:
    """
    Compute all structural metrics for a single sample.
    
    Args:
        positions: Atomic positions [num_atoms, 3]
        box: Box dimensions [3]
        bond_cutoff: Cutoff for C-C bonds
        
    Returns:
        metrics: Dictionary containing all computed metrics
    """
    # Neighbor list
    neighbors = compute_neighbors(positions, box, cutoff=bond_cutoff)
    
    # Coordination numbers
    coord_numbers = compute_coordination_numbers(neighbors)
    
    # Bond angles
    angles = compute_bond_angles(positions, box, neighbors)
    
    # Ring statistics
    ring_stats = compute_ring_statistics(positions, box, neighbors)
    
    # RDF
    r, g_r = compute_rdf(positions, box)
    
    # Aggregate metrics
    # Convert Counter keys to regular ints for JSON serialization
    coord_dist = {int(k): int(v) for k, v in Counter(coord_numbers).items()}
    
    metrics = {
        # Coordination statistics
        'coord_mean': float(np.mean(coord_numbers)),
        'coord_std': float(np.std(coord_numbers)),
        'coord_dist': coord_dist,
        
        # sp hybridization fractions
        'frac_sp': float(np.mean(coord_numbers == 2)),   # sp (linear)
        'frac_sp2': float(np.mean(coord_numbers == 3)),  # sp2 (planar)
        'frac_sp3': float(np.mean(coord_numbers == 4)),  # sp3 (tetrahedral)
        
        # Bond angle statistics
        'angle_mean': float(np.mean(angles)) if len(angles) > 0 else 0.0,
        'angle_std': float(np.std(angles)) if len(angles) > 0 else 0.0,
        
        # Ring statistics
        'ring_counts': ring_stats,
        'ring_total': sum(ring_stats.values()),
        
        # RDF first peak position (nearest neighbor distance)
        'rdf_first_peak': float(r[np.argmax(g_r)]) if len(g_r) > 0 else 0.0,
        
        # Number of atoms
        'n_atoms': len(positions),
    }
    
    return metrics


def evaluate_samples(
    sample_paths: List[str],
    box_size: Optional[np.ndarray] = None,
) -> Dict[str, any]:
    """
    Evaluate a set of generated samples.
    
    Args:
        sample_paths: List of paths to sample files
        box_size: Box dimensions (auto-detected if None)
        
    Returns:
        results: Aggregated evaluation results
    """
    all_metrics = []
    all_rdf_r = None
    all_rdf_g = []
    all_angles = []
    all_coord = []
    
    for path in tqdm(sample_paths, desc="Evaluating samples"):
        try:
            # Read positions
            if path.endswith('.data'):
                positions = read_lammps_positions(path)
                if box_size is None:
                    box = read_box_size(path)
                else:
                    box = box_size
            else:
                raise ValueError(f"Unknown file format: {path}")
            
            if len(positions) == 0:
                print(f"Warning: No atoms found in {path}")
                continue
            
            # Compute metrics
            metrics = compute_all_metrics(positions, box)
            all_metrics.append(metrics)
            
            # Collect for aggregation
            neighbors = compute_neighbors(positions, box)
            all_coord.extend(compute_coordination_numbers(neighbors))
            all_angles.extend(compute_bond_angles(positions, box, neighbors))
            
            # RDF
            r, g_r = compute_rdf(positions, box)
            if all_rdf_r is None:
                all_rdf_r = r
            all_rdf_g.append(g_r)
            
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue
    
    if len(all_metrics) == 0:
        raise ValueError("No valid samples found!")
    
    # Aggregate metrics
    # Convert all_coord to regular ints for Counter
    coord_overall = {int(k): int(v) for k, v in Counter(all_coord).items()}
    
    results = {
        'num_samples': len(all_metrics),
        
        # Coordination
        'coord_mean': float(np.mean([m['coord_mean'] for m in all_metrics])),
        'coord_mean_std': float(np.std([m['coord_mean'] for m in all_metrics])),
        'coord_overall_dist': coord_overall,
        
        # sp fractions
        'frac_sp': float(np.mean([m['frac_sp'] for m in all_metrics])),
        'frac_sp2': float(np.mean([m['frac_sp2'] for m in all_metrics])),
        'frac_sp3': float(np.mean([m['frac_sp3'] for m in all_metrics])),
        
        # Bond angles
        'angle_mean': float(np.mean(all_angles)) if len(all_angles) > 0 else 0.0,
        'angle_std': float(np.std(all_angles)) if len(all_angles) > 0 else 0.0,
        'angle_histogram': np.histogram(all_angles, bins=50, range=(0, 180))[0].tolist() if len(all_angles) > 0 else [],
        
        # Rings
        'ring_counts': aggregate_ring_counts([m['ring_counts'] for m in all_metrics]),
        
        # RDF
        'rdf_r': all_rdf_r.tolist() if all_rdf_r is not None else [],
        'rdf_g_mean': np.mean(all_rdf_g, axis=0).tolist() if len(all_rdf_g) > 0 else [],
        'rdf_g_std': np.std(all_rdf_g, axis=0).tolist() if len(all_rdf_g) > 0 else [],
        'rdf_first_peak': float(np.mean([m['rdf_first_peak'] for m in all_metrics])),
    }
    
    return results


def aggregate_ring_counts(ring_counts_list: List[Dict[int, int]]) -> Dict[str, float]:
    """Aggregate ring counts across samples."""
    total_counts = defaultdict(int)
    for rc in ring_counts_list:
        for size, count in rc.items():
            total_counts[size] += count
    
    # Average per sample
    n_samples = len(ring_counts_list)
    avg_counts = {f"ring_{size}": count / n_samples for size, count in total_counts.items()}
    
    return avg_counts


def compare_distributions(
    gen_results: Dict[str, any],
    ref_results: Dict[str, any],
) -> Dict[str, float]:
    """
    Compare generated vs reference distributions.
    
    Computes:
    - RDF MSE
    - Coordination MAE
    - Angle distribution KL divergence
    
    Returns:
        comparison: Dictionary of comparison metrics
    """
    comparison = {}
    
    # RDF comparison
    if gen_results['rdf_g_mean'] and ref_results['rdf_g_mean']:
        gen_rdf = np.array(gen_results['rdf_g_mean'])
        ref_rdf = np.array(ref_results['rdf_g_mean'])
        
        # Ensure same length
        min_len = min(len(gen_rdf), len(ref_rdf))
        gen_rdf = gen_rdf[:min_len]
        ref_rdf = ref_rdf[:min_len]
        
        comparison['rdf_mse'] = float(np.mean((gen_rdf - ref_rdf) ** 2))
        comparison['rdf_mae'] = float(np.mean(np.abs(gen_rdf - ref_rdf)))
    
    # Coordination comparison
    comparison['coord_mae'] = abs(gen_results['coord_mean'] - ref_results['coord_mean'])
    comparison['coord_diff'] = gen_results['coord_mean'] - ref_results['coord_mean']
    
    # sp fraction differences
    comparison['sp_diff'] = abs(gen_results['frac_sp'] - ref_results['frac_sp'])
    comparison['sp2_diff'] = abs(gen_results['frac_sp2'] - ref_results['frac_sp2'])
    comparison['sp3_diff'] = abs(gen_results['frac_sp3'] - ref_results['frac_sp3'])
    
    # Angle comparison
    comparison['angle_diff'] = abs(gen_results['angle_mean'] - ref_results['angle_mean'])
    
    # RDF first peak
    comparison['first_peak_diff'] = abs(gen_results['rdf_first_peak'] - ref_results['rdf_first_peak'])
    
    return comparison


def plot_comparison(
    gen_results: Dict[str, any],
    ref_results: Dict[str, any],
    output_path: str,
    title: str = "Generated vs Reference",
):
    """
    Create comparison plots.
    
    Args:
        gen_results: Generated sample results
        ref_results: Reference sample results
        output_path: Path to save the plot
        title: Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. RDF comparison
    ax = axes[0, 0]
    if gen_results['rdf_r'] and gen_results['rdf_g_mean']:
        r = np.array(gen_results['rdf_r'])
        gen_g = np.array(gen_results['rdf_g_mean'])
        gen_std = np.array(gen_results['rdf_g_std'])
        ax.plot(r, gen_g, 'b-', label='Generated', linewidth=2)
        ax.fill_between(r, gen_g - gen_std, gen_g + gen_std, alpha=0.3, color='blue')
        
    if ref_results['rdf_r'] and ref_results['rdf_g_mean']:
        r = np.array(ref_results['rdf_r'])
        ref_g = np.array(ref_results['rdf_g_mean'])
        ref_std = np.array(ref_results['rdf_g_std'])
        ax.plot(r, ref_g, 'r--', label='Reference', linewidth=2)
        ax.fill_between(r, ref_g - ref_std, ref_g + ref_std, alpha=0.3, color='red')
    
    ax.set_xlabel('r (Å)', fontsize=12)
    ax.set_ylabel('g(r)', fontsize=12)
    ax.set_title('Radial Distribution Function', fontsize=14)
    ax.legend()
    ax.set_xlim(0, 6)
    
    # 2. Coordination distribution
    ax = axes[0, 1]
    coord_labels = ['1', '2 (sp)', '3 (sp²)', '4 (sp³)', '5+']
    
    def get_coord_fractions(coord_dist):
        total = sum(coord_dist.values())
        fracs = []
        for c in [1, 2, 3, 4]:
            fracs.append(coord_dist.get(c, 0) / total if total > 0 else 0)
        fracs.append(sum(coord_dist.get(c, 0) for c in range(5, 10)) / total if total > 0 else 0)
        return fracs
    
    gen_fracs = get_coord_fractions(gen_results['coord_overall_dist'])
    ref_fracs = get_coord_fractions(ref_results['coord_overall_dist'])
    
    x = np.arange(len(coord_labels))
    width = 0.35
    ax.bar(x - width/2, gen_fracs, width, label='Generated', color='blue', alpha=0.7)
    ax.bar(x + width/2, ref_fracs, width, label='Reference', color='red', alpha=0.7)
    ax.set_xlabel('Coordination Number', fontsize=12)
    ax.set_ylabel('Fraction', fontsize=12)
    ax.set_title('Coordination Distribution', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(coord_labels)
    ax.legend()
    
    # 3. Bond angle distribution
    ax = axes[1, 0]
    angle_bins = np.linspace(0, 180, 51)
    angle_centers = (angle_bins[:-1] + angle_bins[1:]) / 2
    
    if gen_results['angle_histogram']:
        gen_hist = np.array(gen_results['angle_histogram'])
        gen_hist = gen_hist / gen_hist.sum() if gen_hist.sum() > 0 else gen_hist
        ax.plot(angle_centers, gen_hist, 'b-', label='Generated', linewidth=2)
    
    if ref_results['angle_histogram']:
        ref_hist = np.array(ref_results['angle_histogram'])
        ref_hist = ref_hist / ref_hist.sum() if ref_hist.sum() > 0 else ref_hist
        ax.plot(angle_centers, ref_hist, 'r--', label='Reference', linewidth=2)
    
    # Add vertical lines for ideal angles
    ax.axvline(120, color='green', linestyle=':', label='sp² (120°)', alpha=0.7)
    ax.axvline(109.5, color='orange', linestyle=':', label='sp³ (109.5°)', alpha=0.7)
    
    ax.set_xlabel('Bond Angle (°)', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Bond Angle Distribution', fontsize=14)
    ax.legend()
    ax.set_xlim(0, 180)
    
    # 4. Summary metrics
    ax = axes[1, 1]
    ax.axis('off')
    
    comparison = compare_distributions(gen_results, ref_results)
    
    summary_text = f"""
Summary Metrics

                    Generated    Reference    Diff
Coordination:       {gen_results['coord_mean']:.2f}         {ref_results['coord_mean']:.2f}         {comparison['coord_diff']:+.2f}
sp fraction:        {gen_results['frac_sp']*100:.1f}%        {ref_results['frac_sp']*100:.1f}%
sp² fraction:       {gen_results['frac_sp2']*100:.1f}%        {ref_results['frac_sp2']*100:.1f}%        
sp³ fraction:       {gen_results['frac_sp3']*100:.1f}%        {ref_results['frac_sp3']*100:.1f}%
Mean angle:         {gen_results['angle_mean']:.1f}°         {ref_results['angle_mean']:.1f}°         {comparison['angle_diff']:+.1f}°
RDF 1st peak:       {gen_results['rdf_first_peak']:.2f} Å       {ref_results['rdf_first_peak']:.2f} Å       {comparison['first_peak_diff']:+.2f} Å

RDF MSE: {comparison.get('rdf_mse', 0):.4f}
RDF MAE: {comparison.get('rdf_mae', 0):.4f}
"""
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to: {output_path}")


def find_data_files(directory: str, pattern: str = "*.data") -> List[str]:
    """Find all matching data files in directory."""
    path = Path(directory)
    files = list(path.glob(f"**/{pattern}"))
    return [str(f) for f in files]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate amorphous carbon structures"
    )
    
    parser.add_argument(
        '--generated', '-g',
        type=str,
        required=True,
        help='Directory containing generated samples'
    )
    parser.add_argument(
        '--reference', '-r',
        type=str,
        default=None,
        help='Directory containing reference samples (optional)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='evaluation_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--cooling_rate',
        type=int,
        default=None,
        help='Specific cooling rate to evaluate (optional)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to evaluate'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find generated samples
    gen_files = find_data_files(args.generated)
    if args.max_samples:
        gen_files = gen_files[:args.max_samples]
    
    print(f"Found {len(gen_files)} generated samples")
    
    if len(gen_files) == 0:
        print("No generated samples found!")
        return
    
    # Evaluate generated samples
    print("\nEvaluating generated samples...")
    gen_results = evaluate_samples(gen_files)
    
    # Save generated results
    gen_results_path = output_dir / "generated_metrics.json"
    with open(gen_results_path, 'w') as f:
        json.dump(gen_results, f, indent=2)
    print(f"Generated metrics saved to: {gen_results_path}")
    
    # If reference provided, evaluate and compare
    if args.reference:
        ref_files = find_data_files(args.reference)
        if args.max_samples:
            ref_files = ref_files[:args.max_samples]
        
        print(f"\nFound {len(ref_files)} reference samples")
        
        if len(ref_files) > 0:
            print("Evaluating reference samples...")
            ref_results = evaluate_samples(ref_files)
            
            # Save reference results
            ref_results_path = output_dir / "reference_metrics.json"
            with open(ref_results_path, 'w') as f:
                json.dump(ref_results, f, indent=2)
            
            # Compare
            comparison = compare_distributions(gen_results, ref_results)
            comparison_path = output_dir / "comparison.json"
            with open(comparison_path, 'w') as f:
                json.dump(comparison, f, indent=2)
            print(f"Comparison saved to: {comparison_path}")
            
            # Plot comparison
            plot_path = output_dir / "comparison_plot.png"
            plot_comparison(gen_results, ref_results, str(plot_path))
    
    # Print summary
    print(f"\n{'='*50}")
    print("Evaluation Summary")
    print(f"{'='*50}")
    print(f"Samples evaluated: {gen_results['num_samples']}")
    print(f"Mean coordination: {gen_results['coord_mean']:.2f} ± {gen_results['coord_mean_std']:.2f}")
    print(f"sp fraction:  {gen_results['frac_sp']*100:.1f}%")
    print(f"sp² fraction: {gen_results['frac_sp2']*100:.1f}%")
    print(f"sp³ fraction: {gen_results['frac_sp3']*100:.1f}%")
    print(f"Mean bond angle: {gen_results['angle_mean']:.1f}° ± {gen_results['angle_std']:.1f}°")
    print(f"RDF first peak: {gen_results['rdf_first_peak']:.2f} Å")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
