#!/usr/bin/env python3
"""
檢查並加載 DWAQ 模型
展示如何正確使用需要歷史緩衝區的模型
"""
import torch
import torch.nn as nn
import yaml

# 根據 checkpoint 重建網絡結構
class Encoder(nn.Module):
    """VAE Encoder: 將歷史觀察編碼為潛變量和速度估計"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(225, 128),  # 225 = 45 × 5 (5個歷史步)
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU()
        )
        self.encode_mean_latent = nn.Linear(64, 16)
        self.encode_logvar_latent = nn.Linear(64, 16)
        self.encode_mean_vel = nn.Linear(64, 3)
        self.encode_logvar_vel = nn.Linear(64, 3)
    
    def forward(self, x):
        # x shape: (batch, 225) = (batch, 45 × 5)
        features = self.encoder(x)
        latent_mean = self.encode_mean_latent(features)
        vel_mean = self.encode_mean_vel(features)
        return torch.cat([vel_mean, latent_mean], dim=-1)  # (batch, 19)

class Actor(nn.Module):
    """Actor: 根據當前觀察和編碼特徵生成動作"""
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(64, 512),   # 64 = 45(當前觀察) + 19(編碼特徵)
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 12)
        )
    
    def forward(self, x):
        return self.actor(x)

class ActorCriticDWAQ(nn.Module):
    """DWAQ Actor-Critic 模型"""
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.actor = Actor()
        self.std = nn.Parameter(torch.ones(12))
    
    def forward(self, obs_history):
        """
        Args:
            obs_history: (batch, 225) = 5個歷史步 × 45維觀察
                每個時間步 45 維包含：
                - ang_vel (3) + gravity (3) + cmd (3) 
                + joint_pos (12) + joint_vel (12) + action (12)
        Returns:
            action: (batch, 12)
        """
        # 編碼歷史
        encoded_features = self.encoder(obs_history)  # (batch, 19)
        
        # 當前觀察 (最新的45維)
        current_obs = obs_history[:, :45]
        
        # 組合：當前觀察 + 編碼特徵
        actor_input = torch.cat([current_obs, encoded_features], dim=-1)  # (batch, 64)
        
        # 生成動作
        action = self.actor(actor_input)
        return action

def main():
    # 加載配置
    with open('config/big_reddog_dwaq.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    print("=" * 70)
    print("DWAQ 模型配置確認")
    print("=" * 70)
    
    # 加載 checkpoint
    checkpoint = torch.load('pre_train/dwaq.pt', map_location='cpu')
    
    # 創建模型
    model = ActorCriticDWAQ()
    
    # 加載權重
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print(f"\n✓ 模型加載成功！")
    print(f"\n配置參數：")
    print(f"  - num_obs:           {config['num_obs']} (總觀察維度)")
    print(f"  - one_step_obs_size: {config['one_step_obs_size']} (單步觀察)")
    print(f"  - obs_buffer_size:   {config['obs_buffer_size']} (歷史步數)")
    print(f"  - num_actions:       {config['num_actions']}")
    
    print(f"\n觀察空間組成 (單步 {config['one_step_obs_size']} 維)：")
    print(f"  ├─ 角速度 (ang_vel):       3 維")
    print(f"  ├─ 重力投影 (gravity):     3 維")
    print(f"  ├─ 速度指令 (cmd):         3 維")
    print(f"  ├─ 關節位置 (joint_pos):  12 維")
    print(f"  ├─ 關節速度 (joint_vel):  12 維")
    print(f"  └─ 上次動作 (action):     12 維")
    print(f"      ────────────────────────────")
    print(f"      總計:                     45 維")
    
    print(f"\n歷史緩衝區：")
    print(f"  - 需要維護 {config['obs_buffer_size']} 個時間步的觀察歷史")
    print(f"  - 總輸入維度：{config['one_step_obs_size']} × {config['obs_buffer_size']} = {config['num_obs']} 維")
    
    print(f"\n模型架構：")
    print(f"  輸入 (225維歷史觀察)")
    print(f"    ↓")
    print(f"  Encoder (VAE)")
    print(f"    ├─ Linear(225, 128) + ELU")
    print(f"    ├─ Linear(128, 64) + ELU")
    print(f"    └─ 輸出: 19維 (3維速度 + 16維潛變量)")
    print(f"    ↓")
    print(f"  拼接: [當前觀察45維 + 編碼特徵19維] = 64維")
    print(f"    ↓")
    print(f"  Actor MLP")
    print(f"    ├─ Linear(64, 512) + ELU")
    print(f"    ├─ Linear(512, 256) + ELU")
    print(f"    ├─ Linear(256, 128) + ELU")
    print(f"    └─ Linear(128, 12)")
    print(f"    ↓")
    print(f"  輸出 (12維動作)")
    
    # 測試推理
    print(f"\n" + "=" * 70)
    print("測試推理")
    print("=" * 70)
    test_input = torch.randn(1, config['num_obs'])
    with torch.no_grad():
        output = model(test_input)
    print(f"輸入形狀:  {test_input.shape}  (1個樣本, {config['num_obs']}維)")
    print(f"輸出形狀:  {output.shape}  (1個樣本, {config['num_actions']}維)")
    print(f"輸出值範圍: [{output.min():.4f}, {output.max():.4f}]")
    
    print(f"\n" + "=" * 70)
    print("重要提醒")
    print("=" * 70)
    print("✓ 此模型需要歷史緩衝區，不能改為單個 obs")
    print("✓ 在 MuJoCo 模擬中需要維護 obs_tensor_buf")
    print("✓ 緩衝區更新: [新obs, 舊obs[:-45]] -> 保持最新的5個時間步")
    print("=" * 70)

if __name__ == "__main__":
    main()
