"""
Amorphous Carbon Dataset Module for Flow Matching

基于 DM2 的数据加载代码，适配 CrystalFlow 框架。
用于加载 2D 非晶碳的 LAMMPS 数据文件。

冷却速率提取逻辑来自 AMC-FlowGen (Jiani Hu)
"""

import os
import glob
import random
from pathlib import Path
from typing import Optional, Sequence, List, Union, Dict

import numpy as np
import torch
import lightning as pl
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# ASE for reading LAMMPS files and computing neighbor lists
import ase.io
from ase.neighborlist import primitive_neighbor_list


# ============================================================================
# 冷却速率配置 (来自 AMC-FlowGen)
# ============================================================================

# 10种冷却速率 (K/ps)，对应文件索引规则
QUENCHING_RATES = [20, 50, 100, 200, 400, 800, 1500, 2500, 5000, 10000]

# 用于归一化的范围
LOG_RATE_MIN = np.log10(min(QUENCHING_RATES))  # log10(20) ≈ 1.30
LOG_RATE_MAX = np.log10(max(QUENCHING_RATES))  # log10(10000) = 4.0


def get_quenching_rate_from_filename(filepath: Union[str, Path]) -> float:
    """
    从文件名提取冷却速率 (来自 AMC-FlowGen)
    
    文件命名规则: {index}_min.data
    冷却速率规则: (file_index % 100) // 10 → rate_idx (0-9)
    
    示例：
    - 1_min.data  → idx=0, rate_idx=0 → 20 K/ps
    - 11_min.data → idx=10, rate_idx=1 → 50 K/ps
    - 21_min.data → idx=20, rate_idx=2 → 100 K/ps
    - 101_min.data → idx=0, rate_idx=0 → 20 K/ps (循环)
    
    Args:
        filepath: LAMMPS 数据文件路径
        
    Returns:
        冷却速率 (K/ps)
    """
    filepath = Path(filepath)
    # 提取文件名中的数字索引: "123_min.data" → 123
    file_index = int(filepath.stem.split('_')[0])
    
    # 根据规则计算冷却速率索引
    idx = (file_index - 1) % 100
    rate_idx = idx // 10
    
    return float(QUENCHING_RATES[rate_idx])


def normalize_log_rate(log_rate: float) -> float:
    """
    将 log10(冷却速率) 归一化到 [0, 1] 范围
    
    Args:
        log_rate: log10(cooling_rate)
        
    Returns:
        归一化后的值 [0, 1]
    """
    return (log_rate - LOG_RATE_MIN) / (LOG_RATE_MAX - LOG_RATE_MIN)


def denormalize_log_rate(normalized: float) -> float:
    """
    将归一化值还原为 log10(冷却速率)
    
    Args:
        normalized: [0, 1] 范围的归一化值
        
    Returns:
        log10(cooling_rate)
    """
    return normalized * (LOG_RATE_MAX - LOG_RATE_MIN) + LOG_RATE_MIN


def ase_graph(data: Data, cutoff: float) -> Data:
    """
    将原子结构转换为图表示 (来自 DM2)
    
    Args:
        data: PyG Data 对象，包含 pos, cell, pbc, numbers
        cutoff: 邻居搜索的截断半径 (Å)
    
    Returns:
        添加了 edge_index 和 edge_attr 的 Data 对象
    """
    # 获取位置 (可能是 tensor 或 numpy)
    if isinstance(data.pos, torch.Tensor):
        positions = data.pos.numpy()
    else:
        positions = data.pos
    
    # 获取 cell (可能是 ASE Cell 对象或 numpy array)
    if hasattr(data.cell, 'array'):
        cell = data.cell.array
    elif isinstance(data.cell, torch.Tensor):
        cell = data.cell.numpy()
    else:
        cell = np.array(data.cell)
    
    # 使用 ASE 计算周期性邻居列表
    i, j, D = primitive_neighbor_list(
        'ijD',
        cutoff=cutoff,
        pbc=data.pbc,
        cell=cell,
        positions=positions,
        numbers=data.numbers
    )
    
    data.edge_index = torch.tensor(np.stack((i, j)), dtype=torch.long)
    data.edge_attr = torch.tensor(D, dtype=torch.float)
    
    return data


class AmorphousCarbonDataset(Dataset):
    """
    2D 非晶碳数据集
    
    从 LAMMPS 数据文件加载原子结构，转换为 PyG 图格式。
    自动从文件名提取冷却速率。
    
    Args:
        data_dir: LAMMPS 数据文件所在目录
        cutoff: 图构建的截断半径 (Å)
        duplicate: 每个样本复制次数 (数据增强)
        file_pattern: 文件匹配模式
        file_list: 直接指定文件列表 (可选)
        auto_extract_rate: 是否自动从文件名提取冷却速率
        default_cooling_rate: 当无法提取时使用的默认冷却速率 (K/ps)
    """
    
    def __init__(
        self,
        data_dir: str,
        cutoff: float = 5.0,
        duplicate: int = 1,
        file_pattern: str = "*.data",
        file_list: Optional[List[str]] = None,
        auto_extract_rate: bool = True,
        default_cooling_rate: float = 100.0,
    ):
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.cutoff = cutoff
        self.duplicate = duplicate
        self.auto_extract_rate = auto_extract_rate
        self.default_cooling_rate = default_cooling_rate
        
        # 获取文件列表
        if file_list is not None:
            self.files = [self.data_dir / f for f in file_list]
        else:
            self.files = sorted(glob.glob(str(self.data_dir / file_pattern)))
        
        if len(self.files) == 0:
            raise ValueError(f"No files found in {data_dir} with pattern {file_pattern}")
        
        print(f"Found {len(self.files)} LAMMPS data files")
        
        # 加载并预处理所有数据
        self.cached_data = self._load_and_process()
        
        # 数据复制增强
        if duplicate > 1:
            self.cached_data = [d.clone() for d in self.cached_data for _ in range(duplicate)]
        
        print(f"Dataset size after duplication: {len(self.cached_data)}")
        
        # 打印冷却速率统计
        self._print_rate_statistics()
    
    def _print_rate_statistics(self):
        """打印冷却速率分布统计"""
        rates = [10 ** data.cooling_rate.item() for data in self.cached_data[:len(self.files)]]
        unique_rates = sorted(set(rates))
        print(f"Quenching rates found: {[int(r) for r in unique_rates]} K/ps")
        
        rate_counts = {r: rates.count(r) for r in unique_rates}
        for rate, count in sorted(rate_counts.items()):
            print(f"  {int(rate):5d} K/ps: {count:3d} samples")
    
    def _load_and_process(self) -> List[Data]:
        """加载所有文件并转换为 PyG Data 对象"""
        dataset = []
        
        for filepath in self.files:
            filepath = Path(filepath)
            try:
                # 读取 LAMMPS 数据文件
                atoms = ase.io.read(str(filepath), format='lammps-data')
                
                # 提取冷却速率
                if self.auto_extract_rate:
                    try:
                        cooling_rate = get_quenching_rate_from_filename(filepath)
                    except (ValueError, IndexError):
                        cooling_rate = self.default_cooling_rate
                else:
                    cooling_rate = self.default_cooling_rate
                
                # 转换为 PyG Data 对象
                data = self._atoms_to_data(atoms, filepath, cooling_rate)
                
                # 构建图
                data = ase_graph(data, self.cutoff)
                
                dataset.append(data)
                
            except Exception as e:
                print(f"Warning: Failed to load {filepath}: {e}")
                continue
        
        if len(dataset) == 0:
            raise ValueError("No valid data files could be loaded")
        
        return dataset
    
    def _atoms_to_data(self, atoms, filepath: Path, cooling_rate: float) -> Data:
        """将 ASE Atoms 对象转换为 PyG Data 对象"""
        
        # 原子类型 (碳只有一种，编码为 0)
        unique_numbers = np.unique(atoms.numbers)
        if len(unique_numbers) == 1:
            x = torch.zeros(len(atoms), dtype=torch.long)
        else:
            from sklearn.preprocessing import LabelEncoder
            x = torch.tensor(
                LabelEncoder().fit_transform(atoms.numbers),
                dtype=torch.long
            )
        
        # 原子位置 (笛卡尔坐标)
        pos = torch.tensor(atoms.positions, dtype=torch.float)
        
        # 盒子尺寸
        cell = torch.tensor(atoms.cell.array, dtype=torch.float)
        box_lengths = torch.tensor(atoms.cell.lengths(), dtype=torch.float)
        
        # 冷却速率的多种表示
        log_cooling_rate = np.log10(cooling_rate)
        normalized_rate = normalize_log_rate(log_cooling_rate)
        
        # 提取文件索引
        try:
            file_index = int(filepath.stem.split('_')[0])
        except (ValueError, IndexError):
            file_index = -1
        
        data = Data(
            x=x,                              # 原子类型编码
            pos=pos,                          # 笛卡尔坐标
            cell=cell,                        # 盒子矩阵 (3x3)
            box_lengths=box_lengths,          # 盒子长度 [Lx, Ly, Lz]
            pbc=atoms.pbc,                    # 周期性边界条件
            numbers=atoms.numbers,            # 原子序数
            num_atoms=len(atoms),             # 原子数量
            # 冷却速率 - 多种形式
            cooling_rate=torch.tensor([log_cooling_rate], dtype=torch.float),  # log10
            quench_rate=torch.tensor([cooling_rate], dtype=torch.float),        # 原始值
            condition=torch.tensor([normalized_rate], dtype=torch.float),       # 归一化 [0,1]
            # 元数据
            file_index=torch.tensor([file_index], dtype=torch.long),
        )
        
        return data
    
    def __len__(self) -> int:
        return len(self.cached_data)
    
    def __getitem__(self, idx: int) -> Data:
        return self.cached_data[idx].clone()
    
    def get_box_size(self) -> torch.Tensor:
        """获取盒子尺寸 (假设所有样本盒子相同)"""
        return self.cached_data[0].cell.diagonal()
    
    def get_rate_distribution(self) -> Dict[int, int]:
        """获取冷却速率分布"""
        rates = [int(10 ** data.cooling_rate.item()) for data in self.cached_data[:len(self.files)]]
        return {r: rates.count(r) for r in sorted(set(rates))}


def worker_init_fn(id: int):
    """DataLoader worker 初始化函数，确保随机性正确"""
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)


class AmorphousDataModule(pl.LightningDataModule):
    """
    非晶碳数据模块 (PyTorch Lightning)
    
    管理训练/验证/测试数据集的创建和 DataLoader。
    自动从文件名提取冷却速率。
    """
    
    def __init__(
        self,
        data_dir: str,
        cutoff: float = 5.0,
        duplicate: int = 128,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        seed: int = 42,
        auto_extract_rate: bool = True,
        default_cooling_rate: float = 100.0,
    ):
        super().__init__()
        
        self.data_dir = data_dir
        self.cutoff = cutoff
        self.duplicate = duplicate
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.auto_extract_rate = auto_extract_rate
        self.default_cooling_rate = default_cooling_rate
        
        self.train_dataset: Optional[AmorphousCarbonDataset] = None
        self.val_dataset: Optional[AmorphousCarbonDataset] = None
        self.test_dataset: Optional[AmorphousCarbonDataset] = None
    
    def prepare_data(self) -> None:
        """下载或准备数据 (在单进程中执行)"""
        # 数据已存在，无需准备
        pass
    
    def setup(self, stage: Optional[str] = None) -> None:
        """设置数据集 (可能在多进程中执行)"""
        
        # 获取所有文件
        data_path = Path(self.data_dir) / "data"
        all_files = sorted(glob.glob(str(data_path / "*.data")))
        
        if len(all_files) == 0:
            raise ValueError(f"No data files found in {data_path}")
        
        # 按文件名排序后划分
        n_total = len(all_files)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)
        n_test = n_total - n_train - n_val
        
        # 设置随机种子确保可重复性
        random.seed(self.seed)
        indices = list(range(n_total))
        random.shuffle(indices)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        train_files = [Path(all_files[i]).name for i in train_indices]
        val_files = [Path(all_files[i]).name for i in val_indices]
        test_files = [Path(all_files[i]).name for i in test_indices]
        
        print(f"Data split: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")
        
        if stage is None or stage == "fit":
            self.train_dataset = AmorphousCarbonDataset(
                data_dir=data_path,
                cutoff=self.cutoff,
                duplicate=self.duplicate,
                file_list=train_files,
                auto_extract_rate=self.auto_extract_rate,
                default_cooling_rate=self.default_cooling_rate,
            )
            self.val_dataset = AmorphousCarbonDataset(
                data_dir=data_path,
                cutoff=self.cutoff,
                duplicate=1,  # 验证集不需要复制
                file_list=val_files,
                auto_extract_rate=self.auto_extract_rate,
                default_cooling_rate=self.default_cooling_rate,
            )
        
        if stage is None or stage == "test":
            self.test_dataset = AmorphousCarbonDataset(
                data_dir=data_path,
                cutoff=self.cutoff,
                duplicate=1,  # 测试集不需要复制
                file_list=test_files,
                auto_extract_rate=self.auto_extract_rate,
                default_cooling_rate=self.default_cooling_rate,
            )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=worker_init_fn,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=worker_init_fn,
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=worker_init_fn,
        )


# ============== 测试代码 ==============

def test_dataset():
    """测试数据集加载"""
    import sys
    
    # 数据路径
    data_dir = Path(__file__).parent.parent.parent / "data" / "amorphous_carbon" / "data"
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    print(f"Testing dataset from: {data_dir}")
    print("=" * 50)
    
    # 测试单个数据集
    print("\n1. Testing AmorphousCarbonDataset...")
    ds = AmorphousCarbonDataset(
        data_dir=str(data_dir),
        cutoff=5.0,
        duplicate=1,
        auto_extract_rate=True,
    )
    
    print(f"   Dataset size: {len(ds)}")
    
    # 检查第一个样本
    sample = ds[0]
    print(f"   Sample 0:")
    print(f"      - num_atoms: {sample.num_atoms}")
    print(f"      - pos shape: {sample.pos.shape}")
    print(f"      - x (atom types): {sample.x.unique().tolist()}")
    print(f"      - edge_index shape: {sample.edge_index.shape}")
    print(f"      - edge_attr shape: {sample.edge_attr.shape}")
    print(f"      - cell:\n{sample.cell}")
    print(f"      - pbc: {sample.pbc}")
    print(f"      - cooling_rate (log10): {sample.cooling_rate.item():.4f}")
    print(f"      - quench_rate (K/ps): {sample.quench_rate.item():.1f}")
    print(f"      - condition (normalized): {sample.condition.item():.4f}")
    print(f"      - file_index: {sample.file_index.item()}")
    
    # 检查是否是 2D 结构
    z_coords = sample.pos[:, 2]
    is_2d = torch.allclose(z_coords, torch.zeros_like(z_coords), atol=1e-6)
    print(f"      - Is 2D (z=0): {is_2d}")
    
    # 检查边的统计
    num_edges = sample.edge_index.shape[1]
    avg_neighbors = num_edges / sample.num_atoms
    print(f"      - Num edges: {num_edges}")
    print(f"      - Avg neighbors per atom: {avg_neighbors:.2f}")
    
    # 测试冷却速率分布
    print("\n2. Testing cooling rate distribution...")
    rate_dist = ds.get_rate_distribution()
    print(f"   Rate distribution: {rate_dist}")
    
    # 测试数据模块
    print("\n3. Testing AmorphousDataModule...")
    data_dir_parent = Path(__file__).parent.parent.parent / "data" / "amorphous_carbon"
    
    dm = AmorphousDataModule(
        data_dir=str(data_dir_parent),
        cutoff=5.0,
        duplicate=2,  # 少量复制用于测试
        batch_size=4,
        num_workers=0,
        auto_extract_rate=True,
    )
    
    dm.setup("fit")
    
    print(f"   Train dataset size: {len(dm.train_dataset)}")
    print(f"   Val dataset size: {len(dm.val_dataset)}")
    
    # 测试 DataLoader
    print("\n4. Testing DataLoader...")
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    
    print(f"   Batch size: {batch.num_graphs}")
    print(f"   Batch pos shape: {batch.pos.shape}")
    print(f"   Batch edge_index shape: {batch.edge_index.shape}")
    print(f"   Batch num_atoms: {batch.num_atoms}")
    print(f"   Batch cooling_rate (log10): {batch.cooling_rate.tolist()}")
    print(f"   Batch quench_rate (K/ps): {batch.quench_rate.tolist()}")
    print(f"   Batch condition (normalized): {batch.condition.tolist()}")
    
    print("\n" + "=" * 50)
    print("All tests passed!")
    
    return True


if __name__ == "__main__":
    test_dataset()
