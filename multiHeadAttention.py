"""
多头注意力
"""
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# 定义参数
class ModelArgs():
    def __init__(self, max_seq_len=1024, dim=512, n_heads=8, dropout=0.1):
        self.max_seq_len = max_seq_len
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = dropout

class MultiHeadAttention(nn.Module):
    def __init__(self, args, is_causal=False):
        super().__init__()
        assert args.dim % args.n_heads == 0  # 确保dim能够被总头数整除
        self.head_dim = args.dim // args.n_heads # 每个头的维度
        self.is_causal = is_causal
        self.n_heads = args.n_heads

        # 生成 Q K V
        self.wq = nn.Linear(args.dim, self.n_heads*self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_heads*self.head_dim, bias = False)
        self.wv = nn.Linear(args.dim, self.n_heads*self.head_dim, bias=False)
        
        # 输出权重矩阵
        self.wo = nn.Linear(self.n_heads*self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)

        # 生成掩码矩阵
        if is_causal:
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len),float('-inf'))
            mask = torch.triu(mask, diagonal=1)         # 上三角矩阵部分不变 依然是inf 当diagonal参数设置为0时 对角线元素值为inf 下三角矩阵部分设置为0 
            self.register_buffer("mask", mask)          # mask不是模型参数，所以需要存放在缓冲区


    def forward(self, q, k, v):
        batch_size, seqlen, _ = q.size()
        # 先生成Q K V
        xq,xk,xv = self.wq(q), self.wk(k), self.wv(v)
        # 将 Q K V 划分成多个头
        xq = xq.view(batch_size, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(batch_size, seqlen, self.n_heads, self.head_dim)
        xq = xq.transpose(1,2)
        xk = xk.transpose(1,2)
        xv = xv.transpose(1,2)

        # Q*KT/sqrt(head_dim)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        # 添加掩码遮盖
        if self.is_causal:
            assert hasattr(self, 'mask')
            # 这里截取到序列长度，因为有些序列可能比 max_seq_len 短
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
        # softmax
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = self.attn_dropout(scores)
        # Q*KV
        output = torch.matmul(scores, xv)
        # 合并
        output = output.transpose(1, 2).contiguous().view(batch_size, seqlen, -1)

        # 最终投影回残差流
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

# 默认参数
args = ModelArgs()
model = MultiHeadAttention(args, is_causal=True)
input = torch.randn(1, 10, 512)
output = model(input, input, input)
print(output.shape)