import torch
import torch.nn as nn

def attention(query, key, value, dropout=None):
    '''
    args:
        query: 查询值矩阵，形状 (batch_size, num_heads, seq_len, d_k)
        key: 键值矩阵，形状 (batch_size, num_heads, seq_len, d_k)
        value: 真值矩阵，形状 (batch_size, num_heads, seq_len, d_v)
        dropout: Dropout模块（可选），用于正则化注意力权重
    returns:
        output: 加权后的输出，形状 (batch_size, num_heads, seq_len, d_v)
        p_attn: 注意力权重，形状 (batch_size, num_heads, seq_len, seq_len)
    '''
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / d_k ** 0.5
    p_attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    output = torch.matmul(p_attn, value)
    return output, p_attn

def test_attention():
    batch_size = 10
    seq_len = 10
    num_heads = 8 
    dropout = nn.Dropout(0.1)
    d_k = 64
    d_v = 64
    query = torch.randn(batch_size, num_heads, seq_len, d_k)
    key = torch.randn(batch_size, num_heads, seq_len, d_k)
    value = torch.randn(batch_size, num_heads, seq_len, d_v)
    output, p_attn = attention(query, key, value, dropout)
    # 打印输出形状
    print("Output shape:", output.shape)  # 预期: (batch_size, num_heads, seq_len, d_v)
    print("Attention weights shape:", p_attn.shape)  # 预期: (batch_size, num_heads, seq_len, seq_len)

if __name__ == "__main__":
    test_attention()
