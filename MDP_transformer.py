import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear transformations
        query = self.q_linear(query)
        key = self.k_linear(key)
        value = self.v_linear(value)

        # Reshape for multi-head attention
        query = query.view(batch_size, -1, self.num_heads,
                           self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(
            1, 2)
        value = value.view(batch_size, -1, self.num_heads,
                           self.head_dim).transpose(1, 2)

        # print(f"query shape: {query.shape}")
        # print(f"key shape: {key.shape}")
        # print(f"value shape: {value.shape}")

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        # print(attention_weights.shape)

        # Apply attention weights to value
        context = torch.matmul(attention_weights, value)

        # Reshape and concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            self.d_model)

        # Final linear layer
        output = self.out_linear(context)
        return output, attention_weights


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # Self-attention
        src2, attention_weights = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feed-forward network
        src2 = self.ffn(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attention_weights


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers=4, d_model=32, num_heads=8, d_ff=32 * 4,
                 dropout=0.2, base_num=32, patch_num=14):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, num_heads, d_ff, dropout) for _ in
             range(num_layers)])
        self.num_layers = num_layers
        self.patch_num = patch_num
        self.base_num = base_num
        self.fc1 = nn.Linear(196 * d_model, 14 * base_num)
        self.fc2 = nn.Linear(196 * d_model, 14 * base_num)
        self.fc3 = nn.Linear(196 * d_model, base_num * 16)
        self.fc4 = nn.Linear(196 * d_model, base_num * 16)

        # For each base u,v predict and each patch a score matrix 14*14
        # Or predict two 14x1 vector as a matrix decomposition
        # Predict k base vector [u,v], 2k in total

    def forward(self, src, src_mask=None):
        batch_size = src.shape[0]
        src = src.view(batch_size, self.patch_num * self.patch_num, -1)
        attention_weights = []
        for id, layer in enumerate(self.layers):
            src, attn_weights = layer(src, src_mask)
            attention_weights.append(attn_weights)

        # print(src.shape)
        src = src.view(batch_size, -1)  # b, patch_num, d_model
        v_score = self.fc1(src)
        h_score = self.fc2(src)
        h_base = self.fc3(src)
        v_base = self.fc4(src)
        h_base = h_base.view(batch_size, self.base_num, -1)  # B base_num
        # patch_size
        v_base = v_base.view(batch_size, self.base_num, -1)  # B base_num
        # patch_size
        v_score = v_score.view(batch_size, self.base_num, 14)  # B base_num,
        # sqrt(patch_num)
        h_score = h_score.view(batch_size, self.base_num, 14)  # B base_num,
        # sqrt(patch_num)
        # print("Shape of h_base:", h_base.shape)
        # print("Shape of v_base:", v_base.shape)
        # print("Shape of v_score:", v_score.shape)
        # print("Shape of h_score:", h_score.shape)
        base_matrix = torch.einsum('bij,bik->bijk', h_base, v_base)
        score_matrix = torch.einsum('bij,bik->bijk', h_score, v_score)
        score_matrix = score_matrix.view(batch_size, -1, self.base_num)
        base_matrix = base_matrix.view(batch_size, self.base_num, -1)
        # print("score matrix shape", score_matrix.shape)  # B patch_num base_num
        # print("base matrix shape", base_matrix.shape)  # B base_num patch_size
        # patch_size
        normalized_score = torch.softmax(score_matrix, dim=2)
        final_image = torch.einsum('bij,bjk->bik', normalized_score,
                                   base_matrix)
        final_image = final_image.view(batch_size, self.patch_num,
                                       self.patch_num, 16, 16)
        # print(final_image.shape)
        # 先使用 permute 将张量的维度重新排列
        # 我们需要将 (14, 14, 16, 16) 变成 (14, 16, 14, 16)
        final_image = final_image.permute(0, 1, 3, 2, 4)

        # 然后使用 reshape 将张量变成 (14*16, 14*16) 的大小
        final_image = final_image.reshape(batch_size, 14 * 16, 14 * 16)

        return final_image


if __name__ == '__main__':
    t = torch.rand(2, 14 * 14, 32)  # B, patch_num , -1
    # Noise 作为prompt
    MDP_tf = TransformerEncoder()
    MDP_tf(t)
