"""
前馈神经网络
"""
# import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        # 第一层线性层
        self.w1 = nn.Linear(dim, hidden_dim,bias=False)
        # 第二层线性层
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # 定义dropout层 防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.relu(self.w1(x))))
    

# model = MLP(8, 16, 0.1)
# input = torch.randn((10,8))
# output = model(input)
# print(output)