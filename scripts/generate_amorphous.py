#!/usr/bin/env python
"""
Generate Amorphous Carbon Structures using Flow Matching.

This script generates 2D amorphous carbon structures conditioned on cooling rates
using a trained Flow Matching model.

Usage:
    # Generate with default settings
    python scripts/generate_amorphous.py --checkpoint path/to/model.ckpt
    
    # Generate with specific cooling rate
    python scripts/generate_amorphous.py --checkpoint path/to/model.ckpt --cooling_rate 100
    
    # Generate multiple samples
    python scripts/generate_amorphous.py --checkpoint path/to/model.ckpt --num_samples 100
    
    # Generate for all cooling rates
    python scripts/generate_amorphous.py --checkpoint path/to/model.ckpt --all_rates
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from diffcsp.pl_modules.amorphous_flow_module import AmorphousFlowModule

# Standard cooling rates (K/ps)
COOLING_RATES = [20, 50, 100, 200, 400, 800, 1500, 2500, 5000, 10000]


def load_model(checkpoint_path: str, device: str = 'cuda') -> AmorphousFlowModule:
    """Load trained model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get hyperparameters
    hparams = checkpoint.get('hyper_parameters', {})
    
    # Create model
    model = AmorphousFlowModule(**hparams)
    
    # Load weights
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Model type: {hparams.get('model_type', 'nequip')}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def cooling_rate_to_condition(rate: float) -> float:
    """Convert cooling rate to log10 condition value."""
    return np.log10(rate)


def generate_samples(
    model: AmorphousFlowModule,
    num_samples: int,
    cooling_rate: float,
    num_atoms: int = 50,
    steps: int = 100,
    method: str = 'euler',
    device: str = 'cuda',
) -> np.ndarray:
    """
    Generate amorphous carbon samples.
    
    Args:
        model: Trained Flow Matching model
        num_samples: Number of samples to generate
        cooling_rate: Cooling rate in K/ps
        num_atoms: Number of atoms per sample
        steps: Number of ODE integration steps
        method: Integration method ('euler' or 'rk4')
        device: Device to use
        
    Returns:
        positions: Generated atomic positions [num_samples, num_atoms, 3]
    """
    condition = cooling_rate_to_condition(cooling_rate)
    print(f"Generating {num_samples} samples at cooling rate {cooling_rate} K/ps (log10={condition:.3f})")
    
    with torch.no_grad():
        samples, _ = model.sample(
            num_atoms=num_atoms,
            num_samples=num_samples,
            condition=condition,
            steps=steps,
            method=method,
            device=torch.device(device),
            return_trajectory=False,
        )
    
    return samples.cpu().numpy()


def save_lammps(
    positions: np.ndarray,
    output_path: str,
    box_size: List[float] = [12.0, 12.0, 20.0],
    comment: str = "",
):
    """
    Save atomic positions in LAMMPS data format.
    
    Args:
        positions: Atomic positions [num_atoms, 3]
        output_path: Output file path
        box_size: Simulation box size [Lx, Ly, Lz]
        comment: Comment string for header
    """
    num_atoms = len(positions)
    
    with open(output_path, 'w') as f:
        f.write(f"LAMMPS data file - {comment}\n\n")
        f.write(f"{num_atoms} atoms\n")
        f.write("1 atom types\n\n")
        f.write(f"0.0 {box_size[0]} xlo xhi\n")
        f.write(f"0.0 {box_size[1]} ylo yhi\n")
        f.write(f"0.0 {box_size[2]} zlo zhi\n\n")
        f.write("Masses\n\n")
        f.write("1 12.011  # Carbon\n\n")
        f.write("Atoms  # atomic\n\n")
        
        for i, pos in enumerate(positions, 1):
            f.write(f"{i} 1 {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")


def save_xyz(
    positions: np.ndarray,
    output_path: str,
    comment: str = "",
):
    """
    Save atomic positions in XYZ format.
    
    Args:
        positions: Atomic positions [num_atoms, 3]
        output_path: Output file path
        comment: Comment string for header
    """
    num_atoms = len(positions)
    
    with open(output_path, 'w') as f:
        f.write(f"{num_atoms}\n")
        f.write(f"{comment}\n")
        
        for pos in positions:
            f.write(f"C {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")


def generate_and_save(
    model: AmorphousFlowModule,
    output_dir: str,
    cooling_rate: float,
    num_samples: int = 10,
    num_atoms: int = 50,
    steps: int = 100,
    method: str = 'euler',
    format: str = 'lammps',
    device: str = 'cuda',
) -> List[str]:
    """
    Generate samples and save to files.
    
    Returns:
        List of output file paths
    """
    # Create output directory
    rate_dir = Path(output_dir) / f"rate_{int(cooling_rate)}"
    rate_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate samples
    samples = generate_samples(
        model=model,
        num_samples=num_samples,
        cooling_rate=cooling_rate,
        num_atoms=num_atoms,
        steps=steps,
        method=method,
        device=device,
    )
    
    # Save samples
    output_paths = []
    box_size = model.box_size.tolist()
    
    for i, pos in enumerate(samples):
        if format == 'lammps':
            filename = f"gen_{i:04d}.data"
            filepath = rate_dir / filename
            save_lammps(
                pos, filepath, box_size,
                comment=f"Generated sample, cooling_rate={cooling_rate} K/ps"
            )
        elif format == 'xyz':
            filename = f"gen_{i:04d}.xyz"
            filepath = rate_dir / filename
            save_xyz(
                pos, filepath,
                comment=f"Generated sample, cooling_rate={cooling_rate} K/ps"
            )
        else:
            raise ValueError(f"Unknown format: {format}")
        
        output_paths.append(str(filepath))
    
    print(f"Saved {num_samples} samples to {rate_dir}")
    return output_paths


def main():
    parser = argparse.ArgumentParser(
        description="Generate amorphous carbon structures using Flow Matching"
    )
    
    # Required arguments
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='Path to model checkpoint (.ckpt)'
    )
    
    # Generation settings
    parser.add_argument(
        '--cooling_rate', '-r',
        type=float,
        default=None,
        help='Cooling rate in K/ps (default: 100)'
    )
    parser.add_argument(
        '--all_rates',
        action='store_true',
        help='Generate for all standard cooling rates'
    )
    parser.add_argument(
        '--num_samples', '-n',
        type=int,
        default=10,
        help='Number of samples per cooling rate (default: 10)'
    )
    parser.add_argument(
        '--num_atoms',
        type=int,
        default=50,
        help='Number of atoms per sample (default: 50)'
    )
    
    # ODE settings
    parser.add_argument(
        '--steps',
        type=int,
        default=100,
        help='Number of ODE integration steps (default: 100)'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='euler',
        choices=['euler', 'rk4'],
        help='ODE integration method (default: euler)'
    )
    
    # Output settings
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='generated',
        help='Output directory (default: generated)'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='lammps',
        choices=['lammps', 'xyz'],
        help='Output format (default: lammps)'
    )
    
    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (default: cuda if available)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed: {args.seed}")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load model
    model = load_model(args.checkpoint, args.device)
    
    # Determine cooling rates to generate
    if args.all_rates:
        rates = COOLING_RATES
    elif args.cooling_rate is not None:
        rates = [args.cooling_rate]
    else:
        rates = [100]  # Default
    
    # Generate for each cooling rate
    all_paths = []
    for rate in rates:
        paths = generate_and_save(
            model=model,
            output_dir=output_dir,
            cooling_rate=rate,
            num_samples=args.num_samples,
            num_atoms=args.num_atoms,
            steps=args.steps,
            method=args.method,
            format=args.format,
            device=args.device,
        )
        all_paths.extend(paths)
    
    print(f"\n{'='*50}")
    print(f"Generation complete!")
    print(f"Total samples: {len(all_paths)}")
    print(f"Output directory: {output_dir}")
    
    # Save generation config
    config_path = output_dir / "config.txt"
    with open(config_path, 'w') as f:
        f.write(f"Generation Config\n")
        f.write(f"{'='*50}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Cooling rates: {rates}\n")
        f.write(f"Samples per rate: {args.num_samples}\n")
        f.write(f"Atoms per sample: {args.num_atoms}\n")
        f.write(f"ODE steps: {args.steps}\n")
        f.write(f"ODE method: {args.method}\n")
        f.write(f"Format: {args.format}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Seed: {args.seed}\n")
    
    return output_dir


if __name__ == '__main__':
    main()
