"""
层归一化
feature: 样本的维度 这里的样本指代的是 词对应的token
eps：防止分母为0
"""
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, feature, eps=1e-6):
        super().__init__()
        # a_1 a_2 目的是将归一化层后的数据成为可学习的
        self.a_1 = nn.Parameter(torch.ones(feature))
        self.a_2 = nn.Parameter(torch.zeros(feature))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1,keepdim=True)
        return self.a_1*(x - mean)/(std+self.eps)+self.a_2

model = LayerNorm(feature=16)
input = torch.randn((10, 10, 16))
output = model(input)
print(output.shape)