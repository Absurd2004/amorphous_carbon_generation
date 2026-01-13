# CrystalFlow 非晶材料改造 - 实施计划

基于 LAMMPS 数据格式 + DM2 架构，逐步将 CrystalFlow 改造为非晶生成系统。

---

## 前置检查

**验证环境**
```bash
# 1. 检查两个项目都能导入
cd /home/yongkunyang/CrystalFlow && python -c "import diffcsp; print('CrystalFlow OK')"
cd /home/yongkunyang/DM2 && python -c "import graphite; print('DM2 OK')"

# 2. 检查LAMMPS数据文件格式
head -30 /home/yongkunyang/DM2/demo/demo_training/simu_data/sio2_3000_glass_100k_sample0.dat
# 应该看到：原子数、盒子、原子坐标格式
```

---

# 阶段1: 数据层融合 (DM2组件迁移 + Dataset改写)

## 任务1.1: 迁移DM2周期性工具

**文件**: `diffcsp/common/mic.py` (新建)

**步骤**:
```bash
# 1. 复制DM2的MIC模块
cp /home/yongkunyang/DM2/src/graphite/nn/utils/mic.py \
   /home/yongkunyang/CrystalFlow/diffcsp/common/mic.py

# 2. 检查导入依赖
grep -r "import" /home/yongkunyang/DM2/src/graphite/nn/utils/mic.py
# 应该只依赖 torch/numpy

# 3. 调整imports（如果需要）
# - 改 `from graphite.` → `from diffcsp.`
```

**验证**:
```python
# test_mic.py
import torch
from diffcsp.common.mic import minimum_image_convention

pos_i = torch.tensor([[0.0, 0.0, 0.0]])
pos_j = torch.tensor([[0.5, 0.5, 0.5]])
cell = torch.eye(3) * 10.0

delta_mic = minimum_image_convention(pos_i, pos_j, cell)
print(f"Delta MIC shape: {delta_mic.shape}")  # 应该是 [1, 3]
assert delta_mic.shape == (1, 3)
print("✅ MIC测试通过")
```

---

## 任务1.2: 迁移周期性图构建

**文件**: `diffcsp/common/periodic_radius_graph.py` (新建)

**步骤**:
```bash
# 1. 复制DM2的核心函数
cp /home/yongkunyang/DM2/src/graphite/nn/utils/periodic_radius_graph.py \
   /home/yongkunyang/CrystalFlow/diffcsp/common/periodic_radius_graph.py

# 2. 检查依赖
# - mic.py: 已复制
# - torch: 标准库
```

**验证**:
```python
# test_radius_graph.py
import torch
from diffcsp.common.periodic_radius_graph import periodic_radius_graph

pos = torch.rand(100, 3) * 10.0  # 100个原子
cell = torch.eye(3) * 10.0
cutoff = 5.0

edge_index, edge_attr = periodic_radius_graph(pos, cell, r=cutoff)
print(f"边数: {edge_index.shape[1]}")
assert edge_index.shape[0] == 2
assert edge_attr.shape[1] == 3  # [dx, dy, dz]
print("✅ 周期性图构建测试通过")
```

---

## 任务1.3: 迁移边特征计算

**文件**: `diffcsp/common/edges.py` (新建)

**步骤**:
```bash
# 1. 复制DM2的边特征模块
cp /home/yongkunyang/DM2/src/graphite/nn/utils/edges.py \
   /home/yongkunyang/CrystalFlow/diffcsp/common/edges.py
```

**验证**:
```python
# test_edges.py
from diffcsp.common.edges import *  # 导入所有函数
print("✅ 边特征模块导入成功")
```

---

## 任务1.4: 迁移MolData数据结构

**文件**: `diffcsp/pl_data/mol_data.py` (新建)

**步骤**:
```bash
# 1. 复制DM2的MolData
cp /home/yongkunyang/DM2/src/graphite/data/mol.py \
   /home/yongkunyang/CrystalFlow/diffcsp/pl_data/mol_data.py

# 2. 修改文件头部imports
# 改为：from torch_geometric.data import Data
```

**验证**:
```python
# test_mol_data.py
import torch
from diffcsp.pl_data.mol_data import MolData

data = MolData(
    x_atm=torch.tensor([0, 1, 0]),           # 3个原子
    pos=torch.randn(3, 3),
    edge_index=torch.tensor([[0, 1], [1, 2]]),
)
print(f"MolData nodes: {data.x_atm.shape[0]}")
print("✅ MolData测试通过")
```

---

## 任务1.5: 创建配置文件

**文件**: `conf/data/amorphous_carbon.yaml` (新建)

**内容**:
```yaml
# 非晶碳数据集配置
dataset:
  _target_: diffcsp.pl_data.dataset.AmorphousDataset
  root_path: data/amorphous_carbon
  
  # 数据读取
  format: lammps-data      # LAMMPS .dat 格式
  
  # 关闭晶体专用处理
  niggli: false
  primitive: false
  use_space_group: false
  
  # 非晶图构建参数
  graph_method: radius_pbc  # 固定半径图
  cutoff_radius: 5.0        # 碳原子典型值 (Å)
  max_neighbors: 32
  
  # 条件配置
  conditions:
    - name: annealing_rate
      type: continuous
      min: 1e10               # K/s
      max: 1e15
      log_scale: true         # 对数归一化
      embedding_dim: 64

datamodule:
  batch_size:
    train: 16
    val: 16
    test: 16
  num_workers: 4
  pin_memory: true
```

**验证**:
```bash
# test_config.py
from hydra import initialize, compose
from omegaconf import OmegaConf

with initialize(version_base=None, config_path="conf"):
    cfg = compose(config_name="default", overrides=["data=amorphous_carbon"])
    print(OmegaConf.to_yaml(cfg.data))
    assert cfg.data.dataset.cutoff_radius == 5.0
    print("✅ 配置文件测试通过")
```

---

## 任务1.6: 实现AmorphousDataset类

**文件**: `diffcsp/pl_data/dataset.py` (修改 - 添加新类)

**步骤**:

1. **打开现有文件，查看结构**:
```python
# 现有 CrystDataset 的关键方法
# __init__: 初始化
# __getitem__: 返回单个样本
# load_structure: 加载晶体结构
```

2. **添加新类**:
```python
# 在 dataset.py 末尾添加

class AmorphousDataset(CrystDataset):
    """非晶材料数据集"""
    
    def __init__(self, root_path, conditions=None, cutoff_radius=5.0, 
                 format='lammps-data', **kwargs):
        """
        Args:
            root_path: 数据目录
            conditions: 条件字段定义
            cutoff_radius: 图构建半径 (Å)
            format: 数据格式 ('lammps-data')
        """
        # 跳过父类的某些初始化
        self.root_path = Path(root_path)
        self.conditions = conditions or []
        self.cutoff_radius = cutoff_radius
        self.format = format
        
        # 加载CSV文件列表
        self.df_train = pd.read_csv(self.root_path / 'train.csv')
        self.df_val = pd.read_csv(self.root_path / 'val.csv')
        self.df_test = pd.read_csv(self.root_path / 'test.csv')
        self.current_split = 'train'
        self.df = self.df_train
        
    def set_split(self, split='train'):
        """切换数据集分割"""
        if split == 'train':
            self.df = self.df_train
        elif split == 'val':
            self.df = self.df_val
        elif split == 'test':
            self.df = self.df_test
        self.current_split = split
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """读取单个非晶结构"""
        row = self.df.iloc[idx]
        
        # 1. 读取LAMMPS文件
        atoms = ase.io.read(
            self.root_path / 'structures' / row['filename'],
            format=self.format
        )
        
        # 2. 提取坐标和盒子
        pos = torch.FloatTensor(atoms.positions)  # [N, 3] 笛卡尔坐标
        cell = torch.FloatTensor(atoms.cell.array)  # [3, 3]
        atom_types = torch.LongTensor(atoms.numbers)  # [N]
        
        # 3. 构建MolData
        from diffcsp.pl_data.mol_data import MolData
        data = MolData(
            pos=pos,
            x_atm=atom_types,
            cell=cell,
            pbc=torch.BoolTensor([True, True, True]),
        )
        
        # 4. 构建图 (使用DM2的方法)
        from diffcsp.common.periodic_radius_graph import periodic_radius_graph
        edge_index, edge_attr = periodic_radius_graph(
            pos, cell, r=self.cutoff_radius
        )
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        
        # 5. 加载条件
        for cond_cfg in self.conditions:
            name = cond_cfg['name']
            value = row[name]
            
            if cond_cfg.get('log_scale', False):
                value = np.log10(value)
            
            setattr(data, name, torch.FloatTensor([value]))
        
        return data
```

**验证**:
```python
# test_amorphous_dataset.py
from diffcsp.pl_data.dataset import AmorphousDataset

# 需要先有训练数据
dataset = AmorphousDataset(
    root_path='data/amorphous_carbon',
    conditions=[{'name': 'annealing_rate', 'log_scale': True}],
    cutoff_radius=5.0
)

# 检查数据集大小
print(f"Dataset size: {len(dataset)}")
assert len(dataset) > 0, "数据集为空"

# 检查单个样本
sample = dataset[0]
print(f"Sample keys: {sample.keys}")
assert hasattr(sample, 'pos')
assert hasattr(sample, 'edge_index')
assert hasattr(sample, 'annealing_rate')
print("✅ AmorphousDataset测试通过")
```

---

## 任务1.7: 准备示例数据

**文件**: `scripts/prepare_amorphous_data.py` (新建)

**步骤**:

1. **创建数据转换脚本**:
```python
#!/usr/bin/env python3
"""
将DM2的LAMMPS数据转换为CrystalFlow训练格式
"""
import ase.io
import pandas as pd
from pathlib import Path
import numpy as np

def prepare_amorphous_data(
    source_dir='/home/yongkunyang/DM2/demo/demo_training/simu_data',
    target_dir='data/amorphous_carbon',
    train_ratio=0.8,
    val_ratio=0.1,
):
    """
    从DM2的LAMMPS文件准备训练数据
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(exist_ok=True, parents=True)
    
    # 创建结构存储目录
    struct_dir = target_dir / 'structures'
    struct_dir.mkdir(exist_ok=True)
    
    records = []
    
    # 遍历所有.dat文件
    for dat_file in sorted(Path(source_dir).glob('*.dat')):
        # 解析文件名: sio2_3000_glass_100k_sample0.dat
        parts = dat_file.stem.split('_')
        
        # 提取冷却速率
        rate_str = [p for p in parts if 'k' in p.lower()][0]  # e.g., "100k"
        cooling_rate = float(rate_str.replace('k', '')) * 1000  # K/s
        
        # 读取结构
        try:
            atoms = ase.io.read(dat_file, format='lammps-data')
        except Exception as e:
            print(f"Failed to read {dat_file}: {e}")
            continue
        
        # 保存结构副本
        target_file = struct_dir / dat_file.name
        ase.io.write(target_file, atoms, format='lammps-data')
        
        # 计算结构特征
        density = atoms.get_volume() / len(atoms)  # Å³ per atom
        
        record = {
            'filename': dat_file.name,
            'num_atoms': len(atoms),
            'annealing_rate': cooling_rate,  # K/s
            'density': density,
        }
        records.append(record)
    
    # 创建DataFrame
    df = pd.DataFrame(records)
    
    # 划分训练/验证/测试
    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    df_train = df.iloc[:n_train]
    df_val = df.iloc[n_train:n_train+n_val]
    df_test = df.iloc[n_train+n_val:]
    
    # 保存CSV
    df_train.to_csv(target_dir / 'train.csv', index=False)
    df_val.to_csv(target_dir / 'val.csv', index=False)
    df_test.to_csv(target_dir / 'test.csv', index=False)
    
    print(f"✅ 数据准备完成:")
    print(f"   Train: {len(df_train)} samples")
    print(f"   Val:   {len(df_val)} samples")
    print(f"   Test:  {len(df_test)} samples")

if __name__ == '__main__':
    prepare_amorphous_data()
```

2. **运行脚本**:
```bash
cd /home/yongkunyang/CrystalFlow
python scripts/prepare_amorphous_data.py
# 应该生成: data/amorphous_carbon/{train.csv, val.csv, test.csv, structures/}
```

**验证**:
```bash
# 检查生成的文件
ls -la data/amorphous_carbon/
cat data/amorphous_carbon/train.csv | head -5
# 应该看到: filename, num_atoms, annealing_rate, density
```

---

# 阶段2: 图构建层适配

## 任务2.1: 创建批处理图构建函数

**文件**: `diffcsp/common/data_utils.py` (修改 - 添加新函数)

**步骤**:

1. **添加批处理函数**:
```python
# 在 data_utils.py 末尾添加

def build_amorphous_batch_graph(batch, cutoff=5.0, max_neighbors=32):
    """
    为批量非晶结构构建图
    
    Args:
        batch: PyG Batch对象，包含多个MolData
        cutoff: 截断半径 (Å)
        max_neighbors: 最大邻居数
    
    Returns:
        edge_index: [2, E] 边索引 (累加偏移后)
        edge_attr: [E, 3] 边特征 (相对位置向量)
    """
    from diffcsp.common.periodic_radius_graph import periodic_radius_graph
    
    edge_indices = []
    edge_attrs = []
    
    ptr = 0
    for graph_idx in range(batch.num_graphs):
        # 提取当前图的数据
        mask = batch.batch == graph_idx
        pos_i = batch.pos[mask]
        cell_i = batch.cell[graph_idx]
        
        # 构建图
        ei, ea = periodic_radius_graph(pos_i, cell_i, r=cutoff, max_num_neighbors=max_neighbors)
        
        # 累加索引偏移
        ei_shifted = ei + ptr
        edge_indices.append(ei_shifted)
        edge_attrs.append(ea)
        
        ptr += mask.sum().item()
    
    # 拼接所有边
    edge_index_all = torch.cat(edge_indices, dim=1)
    edge_attr_all = torch.cat(edge_attrs, dim=0)
    
    return edge_index_all, edge_attr_all
```

**验证**:
```python
# test_batch_graph.py
import torch
from torch_geometric.data import Batch
from diffcsp.pl_data.mol_data import MolData
from diffcsp.common.data_utils import build_amorphous_batch_graph

# 创建两个样本
data1 = MolData(
    pos=torch.randn(50, 3) * 10,
    cell=torch.eye(3) * 10,
)
data2 = MolData(
    pos=torch.randn(30, 3) * 10,
    cell=torch.eye(3) * 10,
)

# 创建batch
batch = Batch.from_data_list([data1, data2])

# 构建图
edge_index, edge_attr = build_amorphous_batch_graph(batch, cutoff=5.0)

print(f"Batch图 - 边数: {edge_index.shape[1]}")
assert edge_index.shape[0] == 2
assert edge_attr.shape[1] == 3
assert edge_index.max() < batch.pos.shape[0]
print("✅ 批处理图构建测试通过")
```

---

## 任务2.2: 修改DataModule collate函数

**文件**: `diffcsp/pl_data/datamodule.py` (修改 - 修改collate_fn)

**步骤**:

1. **定位collate_fn**:
```bash
grep -n "def collate_fn" /home/yongkunyang/CrystalFlow/diffcsp/pl_data/datamodule.py
```

2. **修改collate_fn以支持非晶**:
```python
def collate_fn(batch):
    """
    自定义collate函数，支持晶体和非晶
    """
    from torch_geometric.data import Batch
    
    # 检测数据类型
    first_sample = batch[0]
    
    if isinstance(first_sample, MolData):
        # 非晶数据：使用DM2的周期性图构建
        batch_data = Batch.from_data_list(batch)
        
        # 重建周期性图
        from diffcsp.common.data_utils import build_amorphous_batch_graph
        edge_index, edge_attr = build_amorphous_batch_graph(
            batch_data, 
            cutoff=5.0  # 从配置中读取
        )
        batch_data.edge_index = edge_index
        batch_data.edge_attr = edge_attr
        
        return batch_data
    else:
        # 晶体数据：使用原有逻辑
        return original_collate_fn(batch)
```

**验证**:
```python
# test_datamodule.py
from diffcsp.pl_data.datamodule import CrystalDataModule
from torch_geometric.loader import DataLoader

datamodule = CrystalDataModule(
    data_root='data/amorphous_carbon',
    batch_size=16,
)

train_loader = datamodule.train_dataloader()
batch = next(iter(train_loader))

print(f"Batch中的图数: {batch.num_graphs}")
assert hasattr(batch, 'edge_index')
assert batch.edge_index.shape[0] == 2
print("✅ DataModule测试通过")
```

---

# 阶段3: GNN架构混合

## 任务3.1: 迁移EGNN模型

**文件**: `diffcsp/pl_modules/egnn_model.py` (新建)

**步骤**:
```bash
# 1. 复制DM2的EGNN实现
cp /home/yongkunyang/DM2/src/graphite/nn/models/egnn.py \
   /home/yongkunyang/CrystalFlow/diffcsp/pl_modules/egnn_model.py

# 2. 检查依赖
grep -r "from graphite" /home/yongkunyang/CrystalFlow/diffcsp/pl_modules/egnn_model.py
# 改为 from diffcsp
```

**验证**:
```python
# test_egnn.py
import torch
from diffcsp.pl_modules.egnn_model import EGNN

model = EGNN(
    in_node_nf=16,
    hidden_nf=64,
    n_layers=4,
)

h = torch.randn(100, 16)  # 100个节点特征
x = torch.randn(100, 3)   # 100个节点坐标
edge_index = torch.randint(0, 100, (2, 500))  # 500条边
edge_attr = torch.randn(500, 3)  # 边特征(相对位置)

h_out, x_out = model(h, x, edge_index, edge_attr)

print(f"输出 - h: {h_out.shape}, x: {x_out.shape}")
assert h_out.shape == (100, 64)
assert x_out.shape == (100, 3)
print("✅ EGNN测试通过")
```

---

## 任务3.2: 迁移径向基函数

**文件**: `diffcsp/pl_modules/basis.py` (新建)

**步骤**:
```bash
# 复制DM2的径向基函数
cp /home/yongkunyang/DM2/src/graphite/nn/basis.py \
   /home/yongkunyang/CrystalFlow/diffcsp/pl_modules/basis.py
```

**验证**:
```python
# test_basis.py
import torch
from diffcsp.pl_modules.basis import bessel

dist = torch.linspace(0.1, 5.0, 100)
radial_basis = bessel(dist, start=0.0, end=5.0, num_basis=16)

print(f"径向基函数输出形状: {radial_basis.shape}")
assert radial_basis.shape == (100, 16)
print("✅ 径向基函数测试通过")
```

---

## 任务3.3: 创建非晶解码器

**文件**: `diffcsp/pl_modules/amorphous_decoder.py` (新建)

**步骤**:

1. **实现解码器类**:
```python
# diffcsp/pl_modules/amorphous_decoder.py

import torch
import torch.nn as nn
from diffcsp.pl_modules.egnn_model import EGNN
from diffcsp.pl_modules.basis import bessel
from torch_scatter import scatter

class AmorphousDecoder(nn.Module):
    """
    非晶材料的Flow Matching解码器
    
    结构:
    - EGNN主干: 消息传递
    - 时间嵌入: Flow时间步
    - 条件嵌入: 退火速率
    - 输出头: 预测速度场
    """
    
    def __init__(
        self,
        hidden_dim=256,
        num_layers=6,
        num_radial=16,
        cutoff=5.0,
        cond_emb_dim=64,
    ):
        super().__init__()
        
        # 1. 时间嵌入 (复用CrystalFlow的设计)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 2. 条件嵌入 (退火速率)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 3. 原子类型嵌入
        self.atom_embedding = nn.Embedding(100, hidden_dim)
        
        # 4. 径向基函数
        self.radial_basis = lambda d: bessel(d, start=0.0, end=cutoff, num_basis=num_radial)
        
        # 5. 边特征投影
        self.edge_mlp = nn.Sequential(
            nn.Linear(num_radial, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 6. EGNN主干
        self.egnn = EGNN(
            in_node_nf=hidden_dim,
            hidden_nf=hidden_dim,
            n_layers=num_layers,
        )
        
        # 7. 速度预测头
        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),  # 输出3D速度向量
        )
    
    def forward(
        self,
        pos,           # [N, 3] 原子坐标
        edge_index,    # [2, E] 边索引
        edge_attr,     # [E, 3] 相对位置向量
        atom_types,    # [N] 原子序数
        time_emb,      # [B, hidden_dim] 时间嵌入
        cond_emb,      # [B, cond_dim] 条件嵌入
        batch=None,    # [N] batch索引
    ):
        """
        预测速度场
        """
        n_atoms = pos.shape[0]
        
        if batch is None:
            batch = torch.zeros(n_atoms, dtype=torch.long, device=pos.device)
        
        # 1. 计算距离
        dist = edge_attr.norm(dim=-1)  # [E]
        
        # 2. 编码边特征
        edge_feat = self.radial_basis(dist)  # [E, num_radial]
        edge_feat = self.edge_mlp(edge_feat)  # [E, hidden_dim]
        
        # 3. 初始节点特征
        h = self.atom_embedding(atom_types)  # [N, hidden_dim]
        
        # 4. 添加时间信息
        time_per_atom = time_emb[batch]  # [N, hidden_dim]
        h = h + time_per_atom
        
        # 5. 添加条件信息
        cond_per_atom = self.cond_mlp(cond_emb)[batch]  # [N, hidden_dim]
        h = h + cond_per_atom
        
        # 6. EGNN传播
        h, pos_update = self.egnn(h, pos, edge_index, edge_feat)
        
        # 7. 预测速度
        velocity = self.velocity_head(h)  # [N, 3]
        
        return {
            'velocity': velocity,
            'pos_update': pos_update,  # EGNN的坐标更新（可选）
        }
```

**验证**:
```python
# test_amorphous_decoder.py
import torch
from diffcsp.pl_modules.amorphous_decoder import AmorphousDecoder

model = AmorphousDecoder(
    hidden_dim=256,
    num_layers=6,
    cond_emb_dim=64,
)

# 构造输入
pos = torch.randn(100, 3)
edge_index = torch.randint(0, 100, (2, 500))
edge_attr = torch.randn(500, 3)
atom_types = torch.randint(0, 6, (100,))  # C, H, O, N等
time_emb = torch.randn(4, 256)  # 4个batch
cond_emb = torch.randn(4, 64)   # 4个条件向量
batch = torch.repeat_interleave(torch.arange(4), 25)  # 25个原子/sample

output = model(
    pos=pos,
    edge_index=edge_index,
    edge_attr=edge_attr,
    atom_types=atom_types,
    time_emb=time_emb,
    cond_emb=cond_emb,
    batch=batch,
)

print(f"速度场输出形状: {output['velocity'].shape}")
assert output['velocity'].shape == (100, 3)
print("✅ 非晶解码器测试通过")
```

---

## 任务3.4: 创建模型配置

**文件**: `conf/model/decoder/egnn_amorphous.yaml` (新建)

**内容**:
```yaml
# EGNN非晶解码器配置
_target_: diffcsp.pl_modules.amorphous_decoder.AmorphousDecoder

hidden_dim: 256
num_layers: 6
num_radial: 16
cutoff: 5.0           # Å
cond_emb_dim: 64      # 条件嵌入维度
```

**验证**:
```bash
cd /home/yongkunyang/CrystalFlow
python -c "
from hydra import initialize, compose
from omegaconf import OmegaConf
with initialize(version_base=None, config_path='conf'):
    cfg = compose(config_name='default', overrides=['model/decoder=egnn_amorphous'])
    print(OmegaConf.to_yaml(cfg.model.decoder))
"
```

---

# 阶段4: Flow核心改造

## 任务4.1: 创建正交晶格工具

**文件**: `diffcsp/pl_modules/lattice_utils.py` (修改 - 添加新类)

**步骤**:

1. **添加OrthogonalLattice类**:
```python
# 在 lattice_utils.py 末尾添加

class OrthogonalLattice:
    """
    正交晶格工具（非晶专用）
    """
    
    @staticmethod
    def from_lengths(Lx, Ly, Lz):
        """从边长构建正交晶格"""
        import torch
        L = torch.tensor([Lx, Ly, Lz])
        return torch.diag(L).unsqueeze(0)
    
    @staticmethod
    def sample_random(batch_size, L_min=8.0, L_max=15.0, device='cpu'):
        """随机初始化晶格（Flow t=1）"""
        import torch
        L = torch.rand(batch_size, 3, device=device) * (L_max - L_min) + L_min
        lattices = torch.diag_embed(L)  # [B, 3, 3]
        return lattices
    
    @staticmethod
    def apply_pbc(pos, cell):
        """
        应用周期性边界条件
        
        Args:
            pos: [N, 3] 原子坐标
            cell: [3, 3] 晶胞矩阵（正交）
        
        Returns:
            pos_wrapped: [N, 3] 映射到[0, L)的坐标
        """
        import torch
        
        # 提取盒子大小（正交情况下是对角元素）
        L = torch.diag(cell)  # [3]
        
        # 映射到[0, L)
        pos_wrapped = pos % L.unsqueeze(0)
        
        return pos_wrapped
    
    @staticmethod
    def interpolate(lattice_0, lattice_1, t):
        """
        线性插值晶格
        
        Args:
            lattice_0: [B, 3, 3] 初始晶格
            lattice_1: [B, 3, 3] 目标晶格
            t: [B] 插值参数（0-1）
        
        Returns:
            lattice_t: [B, 3, 3] 插值晶格
        """
        import torch
        return lattice_0 + t.view(-1, 1, 1) * (lattice_1 - lattice_0)
```

**验证**:
```python
# test_ortho_lattice.py
import torch
from diffcsp.pl_modules.lattice_utils import OrthogonalLattice

# 测试构建
L = OrthogonalLattice.from_lengths(10, 10, 10)
print(f"晶格形状: {L.shape}")
assert L[0, 0, 0] == 10

# 测试随机初始化
L_random = OrthogonalLattice.sample_random(batch_size=4)
print(f"随机晶格形状: {L_random.shape}")
assert L_random.shape == (4, 3, 3)

# 测试PBC
pos = torch.tensor([[10.5, 10.5, 10.5]])
cell = torch.eye(3) * 10
pos_wrapped = OrthogonalLattice.apply_pbc(pos, cell)
print(f"PBC映射后: {pos_wrapped}")
assert (pos_wrapped < 10).all()

print("✅ 正交晶格工具测试通过")
```

---

## 任务4.2: 创建条件嵌入

**文件**: `diffcsp/pl_modules/conditioning.py` (修改 - 添加新类)

**步骤**:

1. **添加退火速率嵌入类**:
```python
# 在 conditioning.py 末尾添加

class AnnealingRateEmbedding(nn.Module):
    """
    退火速率条件嵌入
    """
    
    def __init__(self, output_dim=64):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, output_dim),
        )
    
    def forward(self, annealing_rate):
        """
        Args:
            annealing_rate: [B, 1] 对数归一化的退火速率
        
        Returns:
            emb: [B, output_dim] 条件嵌入
        """
        return self.mlp(annealing_rate)
```

**验证**:
```python
# test_annealing_embedding.py
import torch
from diffcsp.pl_modules.conditioning import AnnealingRateEmbedding

model = AnnealingRateEmbedding(output_dim=64)

# 对数退火速率 (log10(1e12) ≈ 12)
annealing_rate = torch.full((4, 1), 12.0)

emb = model(annealing_rate)
print(f"条件嵌入形状: {emb.shape}")
assert emb.shape == (4, 64)
print("✅ 退火速率嵌入测试通过")
```

---

## 任务4.3: 实现AmorphousFlow模型

**文件**: `diffcsp/pl_modules/amorphous_flow.py` (新建)

**步骤**:

1. **实现Flow模型**:
```python
# diffcsp/pl_modules/amorphous_flow.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from diffcsp.pl_modules.amorphous_decoder import AmorphousDecoder
from diffcsp.pl_modules.conditioning import AnnealingRateEmbedding
from diffcsp.pl_modules.lattice_utils import OrthogonalLattice
from diffcsp.pl_modules.model import TimeEmbedding
from diffcsp.common.data_utils import build_amorphous_batch_graph

class AmorphousFlow(LightningModule):
    """
    非晶材料的Flow Matching生成模型
    """
    
    def __init__(
        self,
        decoder_config,
        cutoff_radius=5.0,
        cost_position=10.0,
        cost_lattice=0.0,
        guide_threshold=0.1,
        ode_int_steps=100,
        **kwargs
    ):
        super().__init__()
        
        # 保存配置
        self.cutoff_radius = cutoff_radius
        self.cost_position = cost_position
        self.cost_lattice = cost_lattice
        self.guide_threshold = guide_threshold
        self.ode_int_steps = ode_int_steps
        
        # 1. 时间嵌入
        self.time_embedding = TimeEmbedding(hidden_dim=256)
        
        # 2. 条件嵌入（退火速率）
        self.cond_embedding = AnnealingRateEmbedding(output_dim=64)
        
        # 3. 解码器（EGNN + Flow头）
        self.decoder = AmorphousDecoder(**decoder_config)
        
        # 4. 正交晶格工具
        self.lattice_tool = OrthogonalLattice()
    
    def forward(self, batch):
        """
        Flow Matching训练步骤
        """
        batch_size = batch.num_graphs
        device = batch.pos.device
        
        # 1. 采样时间步
        t = torch.rand(batch_size, device=device)
        time_emb = self.time_embedding(t)  # [B, 256]
        
        # 2. 条件嵌入
        cond_emb = self.cond_embedding(batch.annealing_rate)  # [B, 64]
        
        # Classifier-free guidance: 随机dropout条件
        if self.guide_threshold is not None and self.training:
            mask = torch.rand(batch_size, device=device) < self.guide_threshold
            cond_emb = torch.where(
                mask.unsqueeze(-1),
                torch.zeros_like(cond_emb),
                cond_emb
            )
        
        # 3. 构建起点(t=0)和终点(t=1)
        # 坐标: 0时刻是盒子内均匀分布
        L_max = batch.cell.diag().max()  # 获取盒子大小
        pos_0 = torch.rand_like(batch.pos) * L_max
        
        # 4. 线性插值到时间t
        t_per_atom = t[batch.batch].unsqueeze(-1)  # [N, 1]
        pos_t = pos_0 + t_per_atom * (batch.pos - pos_0)
        
        # 5. 构建图（重新计算以适应变化的坐标）
        edge_index, edge_attr = build_amorphous_batch_graph(
            batch,
            cutoff=self.cutoff_radius
        )
        
        # 6. 预测速度场
        pred = self.decoder(
            pos=pos_t,
            edge_index=edge_index,
            edge_attr=edge_attr,
            atom_types=batch.x_atm,
            time_emb=time_emb,
            cond_emb=cond_emb,
            batch=batch.batch,
        )
        
        # 7. 计算损失
        target_velocity = batch.pos - pos_0
        loss_pos = F.mse_loss(pred['velocity'], target_velocity)
        
        return loss_pos
    
    def training_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    @torch.no_grad()
    def sample(
        self,
        num_samples=10,
        num_atoms=100,
        annealing_rate=1e12,
        ode_steps=100,
        guide_scale=1.0,
    ):
        """
        ODE积分采样
        """
        device = next(self.parameters()).device
        
        # 1. 初始化 (t=T, 纯噪声)
        L_init = 15.0  # 初始盒子大小
        cell_T = self.lattice_tool.from_lengths(L_init, L_init, L_init).expand(num_samples, -1, -1)
        pos_T = torch.rand(num_samples, num_atoms, 3, device=device) * L_init
        
        # 条件
        cond = torch.full((num_samples, 1), np.log10(annealing_rate), device=device)
        cond_emb = self.cond_embedding(cond)  # [B, 64]
        
        # 2. ODE积分
        pos_t = pos_T
        dt = 1.0 / ode_steps
        trajectory = []
        
        for step in range(ode_steps):
            t_current = 1.0 - step * dt
            times = torch.full((num_samples,), t_current, device=device)
            time_emb = self.time_embedding(times)
            
            # 构建batch进行前向传播
            # ... (构建临时batch)
            
            # 预测速度
            # ... (调用decoder)
            
            # Euler步进
            # pos_t = pos_t - velocity * dt
            
            # 应用PBC
            # pos_t = self.lattice_tool.apply_pbc(pos_t, cell_T)
            
            if step % 10 == 0:
                trajectory.append(pos_t.cpu().numpy())
        
        return pos_t, trajectory
```

**验证**:
```python
# test_amorphous_flow.py
import torch
from diffcsp.pl_modules.amorphous_flow import AmorphousFlow

model = AmorphousFlow(
    decoder_config={'hidden_dim': 256, 'num_layers': 6},
    cutoff_radius=5.0,
)

# 构造假batch
from diffcsp.pl_data.mol_data import MolData
from torch_geometric.data import Batch

data = MolData(
    pos=torch.randn(100, 3) * 10,
    x_atm=torch.zeros(100, dtype=torch.long),
    cell=torch.eye(3) * 10,
    annealing_rate=torch.tensor([12.0]),
)
batch = Batch.from_data_list([data])

# 前向传播
loss = model(batch)
print(f"损失: {loss.item():.4f}")
assert loss.item() > 0
print("✅ AmorphousFlow测试通过")
```

---

## 任务4.4: 创建模型配置

**文件**: `conf/model/amorphous_flow.yaml` (新建)

**内容**:
```yaml
_target_: diffcsp.pl_modules.amorphous_flow.AmorphousFlow

# 解码器配置
decoder_config:
  hidden_dim: 256
  num_layers: 6
  num_radial: 16
  cutoff: 5.0
  cond_emb_dim: 64

# Flow参数
cutoff_radius: 5.0
cost_position: 10.0
cost_lattice: 0.0

# CFG参数
guide_threshold: 0.1

# ODE采样
ode_int_steps: 100
```

---

# 阶段5: 条件生成 + 数据增强

## 任务5.1: 迁移数据增强模块

**文件**: `diffcsp/pl_data/transforms/rattle.py` (新建)

**步骤**:
```bash
# 复制DM2的坐标扰动
cp /home/yongkunyang/DM2/src/graphite/transforms/rattle_particles.py \
   /home/yongkunyang/CrystalFlow/diffcsp/pl_data/transforms/rattle.py
```

**验证**:
```python
# test_rattle.py
import torch
from diffcsp.pl_data.transforms.rattle import RattleParticles

rattle = RattleParticles(sigma_max=0.1)

pos = torch.randn(100, 3)
pos_rattle = rattle(pos)

assert (pos_rattle != pos).any()
assert (pos_rattle.abs() - pos.abs()).abs().max() < 0.2
print("✅ 坐标扰动测试通过")
```

---

## 任务5.2: 修改DataModule集成增强

**文件**: `diffcsp/pl_data/datamodule.py` (修改)

**步骤**:

在train_dataloader中添加transform:

```python
def train_dataloader(self):
    """训练数据加载器（带增强）"""
    from diffcsp.pl_data.transforms.rattle import RattleParticles
    
    loader = DataLoader(
        self.train_dataset,
        batch_size=self.batch_size['train'],
        shuffle=True,
        collate_fn=self.collate_fn,
        num_workers=self.num_workers,
    )
    
    # 添加数据增强wrapper
    if hasattr(self, 'rattle'):
        # apply rattle during loading
        pass
    
    return loader
```

---

## 任务5.3: 创建训练脚本

**文件**: `scripts/train_amorphous.sh` (新建)

**内容**:
```bash
#!/bin/bash
# 非晶材料Flow Matching训练脚本

CUDA_VISIBLE_DEVICES=0,1,2,3 python diffcsp/run.py \
  data=amorphous_carbon \
  data.train_max_epochs=5000 \
  model=amorphous_flow \
  model.cutoff_radius=5.0 \
  model.cost_position=10.0 \
  model.cost_lattice=0.0 \
  +model.guide_threshold=-1 \
  optim.optimizer.lr=5e-4 \
  optim.optimizer.weight_decay=1e-6 \
  optim.lr_scheduler.factor=0.6 \
  train.pl_trainer.devices=4 \
  +train.pl_trainer.strategy=ddp_find_unused_parameters_true \
  +train.pl_trainer.gradient_clip_val=1.0 \
  logging.wandb.mode=online \
  logging.wandb.project=amorphous-carbon-flow \
  expname=AmorphousCarbon-Flow-v1 \
  > train_amorphous.log 2>&1 &

echo "Training started. Check train_amorphous.log for progress."
```

---

## 任务5.4: 创建采样脚本

**文件**: `scripts/sample_amorphous.py` (新建)

**步骤**:

```python
#!/usr/bin/env python3
"""
从训练好的非晶Flow模型采样结构
"""

import torch
import argparse
from pathlib import Path
from diffcsp.pl_modules.amorphous_flow import AmorphousFlow
import numpy as np

def sample_amorphous(
    model_path,
    num_samples=10,
    annealing_rates=[1e10, 1e12, 1e15],
    output_dir='generated_structures',
):
    """
    采样非晶结构
    """
    # 加载模型
    model = AmorphousFlow.load_from_checkpoint(model_path)
    model.eval()
    device = next(model.parameters()).device
    
    # 创建输出目录
    Path(output_dir).mkdir(exist_ok=True)
    
    # 为每个退火速率采样
    for rate in annealing_rates:
        print(f"采样退火速率 {rate:.1e} K/s...")
        
        structures, _ = model.sample(
            num_samples=num_samples,
            num_atoms=500,
            annealing_rate=rate,
            ode_steps=100,
        )
        
        # 保存为XYZ格式
        for i, struct in enumerate(structures):
            output_file = Path(output_dir) / f'amorphous_rate{rate:.0e}_sample{i:03d}.xyz'
            # ... (保存逻辑)
            print(f"  保存到 {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='模型checkpoint路径')
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--output', default='generated_structures')
    args = parser.parse_args()
    
    sample_amorphous(
        model_path=args.model,
        num_samples=args.samples,
        output_dir=args.output,
    )
```

---

# 完整验证清单

## 所有验证都通过后的检查

```bash
# 1. 环境检查
python -c "import diffcsp; import graphite; print('✅ 环境就绪')"

# 2. 数据检查
ls data/amorphous_carbon/{train.csv,val.csv,test.csv} && echo "✅ 数据准备完成"

# 3. 数据加载检查
python -c "
from diffcsp.pl_data.dataset import AmorphousDataset
ds = AmorphousDataset('data/amorphous_carbon')
print(f'✅ 数据集加载成功: {len(ds)} samples')
"

# 4. 模型检查
python -c "
from diffcsp.pl_modules.amorphous_flow import AmorphousFlow
model = AmorphousFlow({'hidden_dim': 256, 'num_layers': 4})
print('✅ 模型初始化成功')
"

# 5. 训练前准备检查
python -c "
from hydra import initialize, compose
with initialize(version_base=None, config_path='conf'):
    cfg = compose(config_name='default', overrides=['data=amorphous_carbon', 'model=amorphous_flow'])
    print(f'✅ 配置加载成功: {cfg.data.dataset.cutoff_radius}Å cutoff')
"
```

## 逐步测试清单

| 任务 | 验证命令 | 预期结果 |
|------|---------|---------|
| 1.1 MIC迁移 | `python test_mic.py` | ✅ MIC测试通过 |
| 1.2 图构建 | `python test_radius_graph.py` | ✅ 周期性图构建测试通过 |
| 1.3 边特征 | `python test_edges.py` | ✅ 边特征模块导入成功 |
| 1.4 MolData | `python test_mol_data.py` | ✅ MolData测试通过 |
| 1.5 配置 | `python test_config.py` | ✅ 配置文件测试通过 |
| 1.6 Dataset | `python test_amorphous_dataset.py` | ✅ AmorphousDataset测试通过 |
| 1.7 数据准备 | `python scripts/prepare_amorphous_data.py` | ✅ 数据准备完成 |
| 2.1 批图构建 | `python test_batch_graph.py` | ✅ 批处理图构建测试通过 |
| 2.2 DataModule | `python test_datamodule.py` | ✅ DataModule测试通过 |
| 3.1 EGNN | `python test_egnn.py` | ✅ EGNN测试通过 |
| 3.2 径向基 | `python test_basis.py` | ✅ 径向基函数测试通过 |
| 3.3 解码器 | `python test_amorphous_decoder.py` | ✅ 非晶解码器测试通过 |
| 3.4 模型配置 | `python -c "..."` | ✅ 配置加载成功 |
| 4.1 晶格工具 | `python test_ortho_lattice.py` | ✅ 正交晶格工具测试通过 |
| 4.2 条件嵌入 | `python test_annealing_embedding.py` | ✅ 退火速率嵌入测试通过 |
| 4.3 Flow模型 | `python test_amorphous_flow.py` | ✅ AmorphousFlow测试通过 |
| 5.1 数据增强 | `python test_rattle.py` | ✅ 坐标扰动测试通过 |

---

## 最终集成测试

```bash
# 运行完整的小规模训练测试
python diffcsp/run.py \
  data=amorphous_carbon \
  data.train_max_epochs=2 \
  model=amorphous_flow \
  +trainer.fast_dev_run=5 \
  2>&1 | tee integration_test.log

# 检查日志
grep -i "error" integration_test.log || echo "✅ 集成测试通过"
```

---

## 故障排查

### 如果数据加载失败:
```bash
# 检查数据文件
ls -la data/amorphous_carbon/structures/ | head -5

# 验证LAMMPS格式
head -20 data/amorphous_carbon/structures/sio2_3000_glass_100k_sample0.dat
```

### 如果模型初始化失败:
```bash
# 逐个检查依赖
python -c "from diffcsp.pl_modules.egnn_model import EGNN; print('EGNN OK')"
python -c "from diffcsp.pl_modules.amorphous_decoder import AmorphousDecoder; print('Decoder OK')"
```

### 如果训练崩溃:
```bash
# 检查梯度
python -c "
import torch
from diffcsp.pl_modules.amorphous_flow import AmorphousFlow
model = AmorphousFlow({'hidden_dim': 256, 'num_layers': 4})
# ... 构造输入
# loss.backward()
# print(model.decoder.egnn.layers[0].weight.grad)
"
```

---

## 下一步

所有验证都通过后，可以开始:
1. **全量数据准备**: 从DM2的所有LAMMPS文件生成训练集
2. **超参数调优**: 调整 `cutoff_radius`, `hidden_dim`, `num_layers` 等
3. **长周期训练**: 运行完整的5000 epochs训练
4. **生成质量评估**: 计算RDF、配位数等指标
5. **条件控制性测试**: 验证不同退火速率生成结构的差异
