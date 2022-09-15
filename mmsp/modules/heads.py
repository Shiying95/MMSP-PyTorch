import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import BertPredictionHeadTransform


class Pooler(nn.Module):
    def __init__(self, hidden_size, pool_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, pool_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]  # 取cls token作为token_tensor
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class MeanPooler(nn.Module):
    def __init__(self, hidden_size, pool_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, pool_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = torch.mean(hidden_states, dim=1)
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class MaxPooler(nn.Module):
    def __init__(self, hidden_size, pool_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, pool_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = torch.max(hidden_states, dim=1)
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class MyPooler(nn.Module):
    def __init__(self, hidden_size, pool_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, pool_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, mask):
        pooled_output = self.dense(hidden_states)  # (None, n, hidden_size) -> (None, n, pool_size)
        pooled_output = self.activation(pooled_output)
        mask = torch.unsqueeze(mask, -1)
        pooled_output = torch.mul(pooled_output, mask)
        pooled_output = torch.flatten(pooled_output, 1, 2)  # (None, n*pool_size)
        return pooled_output


class ITMHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x


class MLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)  # 解码为vocab上的概率分布
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight  # 如果有初始权重传入，则将decoder权重进行初始化

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x


class MPPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, 256 * 3)

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x)
        return x


# class TITHead(nn.Module):
#     def __init__(self, config):
#         super().__init__()
        # self.classifier = nn.Sequential(
        #     nn.Linear(),
        #     nn.LayerNorm(),
        #     nn.GELU(),
        #     nn.Linear(),
        # )

#     def forward(self, x):
#         x = self.classifier(x)
#         return x