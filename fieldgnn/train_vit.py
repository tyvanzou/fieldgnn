import os
import yaml
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning.pytorch import Trainer, loggers
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
)
from typing import Tuple, Dict, List


def set_scheduler(pl_module):
    optim_config = pl_module.config['optimize']
    lr = optim_config["lr"]
    wd = optim_config["weight_decay"]

    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
    ]
    backbone_names = ["head"]
    end_lr = optim_config["end_lr"]
    lr_mult = optim_config["lr_mult"]
    decay_power = optim_config["decay_power"]
    optim_type = optim_config["optim_type"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
            ],
            "weight_decay": wd,
            "lr": lr,
        },
    ]

    if optim_type == "adamw":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.95)
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)

    if pl_module.trainer.max_steps == -1:
        max_steps = pl_module.trainer.estimated_stepping_batches
    else:
        max_steps = pl_module.trainer.max_steps

    warmup_steps = optim_config["warmup_steps"]
    if isinstance(optim_config["warmup_steps"], float):
        warmup_steps = int(max_steps * warmup_steps)

    print(
        f"max_epochs: {pl_module.trainer.max_epochs} | max_steps: {max_steps} | warmup_steps : {warmup_steps} "
        f"| weight_decay : {wd} | decay_power : {decay_power}"
    )

    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    elif decay_power == "constant":
        scheduler = get_constant_schedule(
            optimizer,
        )
    elif decay_power == "constant_with_warmup":
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )

    sched = {"scheduler": scheduler, "interval": "step"}

    return (
        [optimizer],
        [sched],
    )

class PESDataModule(L.LightningDataModule):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.patch_size = config['data']['patch_size']
        self.grid_size = config['data']['grid_size']
        self.train_mean = None
        self.train_std = None
        
    def prepare_data(self):
        train_df = pd.read_csv(os.path.join(self.config['data']['root_dir'], 'benchmark.train.csv'))
        self.train_mean = train_df[self.config['train']['task_name']].mean()
        self.train_std = train_df[self.config['train']['task_name']].std()
    
    def setup(self, stage: str = None):
        normalize_stats = {'mean': self.train_mean, 'std': self.train_std}
        
        if stage == 'fit' or stage is None:
            self.train_dataset = PESDataset(
                os.path.join(self.config['data']['root_dir'], 'benchmark.train.csv'),
                self.config['train']['task_name'],
                self.config['data']['root_dir'],
                grid_size = self.grid_size,
                patch_size = self.patch_size,
                normalize=normalize_stats
            )
            self.val_dataset = PESDataset(
                os.path.join(self.config['data']['root_dir'], 'benchmark.val.csv'),
                self.config['train']['task_name'],
                self.config['data']['root_dir'],
                grid_size = self.grid_size,
                patch_size = self.patch_size,
                normalize=normalize_stats
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = PESDataset(
                os.path.join(self.config['data']['root_dir'], 'benchmark.test.csv'),
                self.config['train']['task_name'],
                self.config['data']['root_dir'],
                grid_size = self.grid_size,
                patch_size = self.patch_size,
                normalize=normalize_stats
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config['train']['batch_size'],
            shuffle=True,
            num_workers=4
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config['train']['batch_size'],
            num_workers=4
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config['train']['batch_size'],
            num_workers=4
        )

class PESDataset(Dataset):
    def __init__(self, csv_path: str, task_name: str, data_root: str, grid_size: int, patch_size: int, normalize: Dict = None):
        self.df = pd.read_csv(csv_path)
        self.data_root = data_root
        self.normalize = normalize
        self.grid_size = grid_size
        self.patch_size = patch_size
        assert grid_size % patch_size == 0
        self.patch_num = self.grid_size // self.patch_size
        self.task_name = task_name
        self.df = self.df.dropna(subset=[self.task_name])
        self.matids = self.df['matid'].tolist()  # 显式存储matid列表
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float, str]:
        matid = self.matids[idx]  # 直接通过索引获取
        label = self.df.iloc[idx][self.task_name]
        
        # 加载并处理PES数据
        pes_data = np.load(os.path.join(self.data_root, 'pes', f'{matid}.npy'))

        pes_data = np.clip(pes_data, -5000, 5000)
        pes_data /= 5000
        
        # 使用einops处理 [30,30,30] -> [10,10,10,3,3,3] -> [1000,27]
        patches = rearrange(pes_data, '(p1 h) (p2 w) (p3 d) -> (p1 p2 p3) (h w d)',
                          p1=self.patch_num, p2=self.patch_num, p3=self.patch_num, h=self.patch_size, w=self.patch_size, d=self.patch_size)
        
        # 添加CLS token [1001,27]
        patches = torch.tensor(patches, dtype=torch.float32)
        cls_token = torch.zeros(1, patches.shape[1])
        tokens = torch.cat([cls_token, patches], dim=0)
        
        # 归一化标签
        label = torch.tensor(label, dtype=torch.float32)
        if self.normalize:
            label = (label - self.normalize['mean']) / self.normalize['std']
            
        return tokens, label, matid  # 明确返回matid

class ViTRegressor(L.LightningModule):
    def __init__(self, config: Dict):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.patch_size = config['data']['patch_size']
        self.grid_size = config['data']['grid_size']
        assert self.grid_size % self.patch_size == 0
        self.patch_num = self.grid_size // self.patch_size
        
        # 模型架构
        input_dim = self.patch_size**3
        dim = config['model']['dim']
        self.patch_proj = nn.Linear(input_dim, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.patch_num**3 + 1, dim))
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=dim,
                nhead=config['model']['heads'],
                dim_feedforward=dim*4,
                dropout=config['model']['dropout'],
                batch_first=True
            ) for _ in range(config['model']['depth'])
        ])
        
        self.regressor = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim//2),
            nn.ReLU(),
            nn.Linear(dim//2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        
        # 投影patches
        x = self.patch_proj(x)
        x = x + self.pos_embedding
        
        # 通过Transformer层
        for layer in self.encoder_layers:
            x = layer(x)

        # 获取最后一层CLS token的注意力权重 [batch, nhead, 1001, 1001]
        attn_weights = self.encoder_layers[-1].self_attn(x, x, x, need_weights=True, average_attn_weights=False)[1]
        cls_attn = attn_weights[:, :, 0, 1:].mean(dim=1)  # 平均多头注意力 [batch, 1000]
        
        return self.regressor(x[:, 0, :]).squeeze(-1), cls_attn
    
    def _save_attention_stats(self, 
                            tokens: torch.Tensor, 
                            attn_scores: torch.Tensor, 
                            matids: List[str]) -> None:
        """保存注意力统计信息"""
        # if not hasattr(self.logger, 'log_dir'):
        #     return
            
        attn_dir = os.path.join(self.logger.log_dir, 'attn')
        os.makedirs(attn_dir, exist_ok=True)
        
        # 处理每个样本
        for i, matid in enumerate(matids):
            # 获取patch数据 [1000,27] (排除CLS token)
            patch_data = tokens[i, 1:].detach().cpu().numpy()
            
            # 计算每个patch的统计量 [1000,3]
            patch_stats = np.column_stack([
                patch_data.mean(axis=1),  # mean
                patch_data.max(axis=1),   # max
                patch_data.min(axis=1)    # min
            ])
            
            # 合并注意力分数 [1000,4]
            combined = np.column_stack([
                patch_stats,
                attn_scores[i].detach().cpu().numpy().reshape(-1, 1)
            ])

            spatial_attn = combined.reshape(self.patch_num, self.patch_num, self.patch_num, 4)
            
            # # 转换为空间结构 [10,10,10,4]
            # spatial_attn = np.zeros((10, 10, 10, 4))
            # for idx in range(1000):
            #     x, y, z = idx//100, (idx%100)//10, idx%10
            #     spatial_attn[x, y, z] = combined[idx]
            
            # 保存为.npy文件
            np.save(os.path.join(attn_dir, f'{matid}.npy'), spatial_attn)
    
    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        tokens, labels, matids = batch
        pred, attn_scores = self(tokens)

        loss = nn.MSELoss()(pred, labels)
        self.log('train_loss', loss, prog_bar=True)
        
        if batch_idx % self.config['train'].get('log_interval', 10) == 0:
            self._save_attention_stats(tokens, attn_scores, matids)
        
        return loss
    
    def validation_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        tokens, labels, matids = batch
        pred, attn_scores = self(tokens)
        self._save_attention_stats(tokens, attn_scores, matids)
        
        loss = nn.MSELoss()(pred, labels)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        tokens, labels, matids = batch
        pred, attn_scores = self(tokens)
        self._save_attention_stats(tokens, attn_scores, matids)
        
        loss = nn.MSELoss()(pred, labels)
        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return set_scheduler(self)
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.config['train']['lr'])
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, 
        #     mode='min', 
        #     patience=3
        # )
        # return {
        #     'optimizer': optimizer,
        #     'lr_scheduler': {
        #         'scheduler': scheduler,
        #         'monitor': 'val_loss'
        #     }
        # }

def main(config_path: str) -> None:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    dm = PESDataModule(config)
    model = ViTRegressor(config)
    
    logger = loggers.TensorBoardLogger(
        save_dir=config['train']['log_dir'],
        name='ViT_Regressor'
    )
    
    # checkpoint_callback = ModelCheckpoint(
    #     monitor='val_loss',
    #     dirpath=os.path.join(config['train']['log_dir'], 'checkpoints'),
    #     filename='best_model',
    #     save_top_k=1,
    #     mode='min'
    # )
    
    trainer = Trainer(
        max_epochs=config['train']['epochs'],
        logger=logger,
        # callbacks=[checkpoint_callback],
        accelerator='gpu' if torch.cuda.is_available() else 'auto',
        devices=config['train']['devices'],
        log_every_n_steps=10
    )
    
    trainer.fit(model, datamodule=dm)
    # trainer.test(model, datamodule=dm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml')
    args = parser.parse_args()
    
    main(args.config)
