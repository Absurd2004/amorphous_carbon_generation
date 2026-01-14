#!/usr/bin/env python3
"""
数据预处理脚本: 准备 2D 非晶碳数据集

功能:
1. 读取所有 LAMMPS 数据文件
2. 验证数据格式和一致性
3. 划分 train/val/test 数据集
4. 生成数据统计信息

使用方法:
    python scripts/prepare_amorphous_carbon.py
"""

import os
import sys
import glob
import json
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import ase.io

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare amorphous carbon dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "amorphous_carbon" / "data"),
        help="Directory containing LAMMPS data files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "amorphous_carbon"),
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Training set ratio"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.1, help="Validation set ratio"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for data split"
    )
    parser.add_argument(
        "--cooling_rate", type=float, default=100.0, help="Default cooling rate (K/ps)"
    )
    return parser.parse_args()


def load_and_validate_structure(filepath: str) -> dict:
    """
    加载并验证单个 LAMMPS 数据文件
    
    返回结构信息字典，如果加载失败返回 None
    """
    try:
        atoms = ase.io.read(filepath, format="lammps-data")
        
        # 提取基本信息
        info = {
            "filepath": filepath,
            "filename": Path(filepath).name,
            "num_atoms": len(atoms),
            "elements": list(set(atoms.get_chemical_symbols())),
            "cell": atoms.cell.array.tolist(),
            "pbc": atoms.pbc.tolist(),
            "positions_min": atoms.positions.min(axis=0).tolist(),
            "positions_max": atoms.positions.max(axis=0).tolist(),
        }
        
        # 检查是否是 2D 结构 (z 坐标是否为 0)
        z_coords = atoms.positions[:, 2]
        info["is_2d"] = np.allclose(z_coords, 0, atol=1e-6)
        info["z_range"] = [z_coords.min(), z_coords.max()]
        
        return info
        
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Amorphous Carbon Dataset Preparation")
    print("=" * 60)
    
    # 查找所有数据文件
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    files = sorted(glob.glob(str(data_dir / "*.data")))
    print(f"\nFound {len(files)} data files in {data_dir}")
    
    if len(files) == 0:
        print("Error: No data files found!")
        sys.exit(1)
    
    # 加载并验证所有结构
    print("\nValidating structures...")
    structures = []
    errors = []
    
    for filepath in files:
        info = load_and_validate_structure(filepath)
        if info is not None:
            structures.append(info)
        else:
            errors.append(filepath)
    
    print(f"Successfully loaded: {len(structures)}")
    print(f"Failed to load: {len(errors)}")
    
    if len(structures) == 0:
        print("Error: No valid structures found!")
        sys.exit(1)
    
    # 统计信息
    print("\n" + "-" * 40)
    print("Dataset Statistics:")
    print("-" * 40)
    
    num_atoms_list = [s["num_atoms"] for s in structures]
    print(f"Number of atoms: {min(num_atoms_list)} - {max(num_atoms_list)} (mean: {np.mean(num_atoms_list):.1f})")
    
    elements_set = set()
    for s in structures:
        elements_set.update(s["elements"])
    print(f"Elements: {sorted(elements_set)}")
    
    is_2d_list = [s["is_2d"] for s in structures]
    print(f"2D structures (z=0): {sum(is_2d_list)} / {len(is_2d_list)}")
    
    # 盒子尺寸
    cells = np.array([s["cell"] for s in structures])
    print(f"Cell shape: {cells[0]}")
    
    # 划分数据集
    print("\n" + "-" * 40)
    print("Data Split:")
    print("-" * 40)
    
    random.seed(args.seed)
    indices = list(range(len(structures)))
    random.shuffle(indices)
    
    n_total = len(structures)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)
    n_test = n_total - n_train - n_val
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    print(f"Train: {n_train} samples ({100 * args.train_ratio:.0f}%)")
    print(f"Val:   {n_val} samples ({100 * args.val_ratio:.0f}%)")
    print(f"Test:  {n_test} samples ({100 * (1 - args.train_ratio - args.val_ratio):.0f}%)")
    
    # 保存划分信息
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    split_info = {
        "train": [structures[i]["filename"] for i in train_indices],
        "val": [structures[i]["filename"] for i in val_indices],
        "test": [structures[i]["filename"] for i in test_indices],
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "cooling_rate": args.cooling_rate,
    }
    
    split_file = output_dir / "data_split.json"
    with open(split_file, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"\nSaved data split to: {split_file}")
    
    # 保存完整统计信息
    stats = {
        "total_samples": len(structures),
        "num_atoms": {
            "min": min(num_atoms_list),
            "max": max(num_atoms_list),
            "mean": float(np.mean(num_atoms_list)),
        },
        "elements": sorted(elements_set),
        "is_2d": sum(is_2d_list) == len(is_2d_list),
        "cell": structures[0]["cell"],
        "pbc": structures[0]["pbc"],
        "split": {
            "train": n_train,
            "val": n_val,
            "test": n_test,
        },
        "cooling_rate": args.cooling_rate,
    }
    
    stats_file = output_dir / "dataset_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved dataset statistics to: {stats_file}")
    
    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
