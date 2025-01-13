import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Parameters
SEQ_LEN = 5
D_MODEL = 4

# Sample Sequence Dataset
sequence = torch.tensor([[1.0, 0.0, 1.0, 0.0],
                         [0.0, 2.0, 0.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0],
                         [0.0, 0.0, 2.0, 1.0],
                         [1.0, 2.0, 0.0, 0.0]])

# Self-Attention Components
class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(K.size(-1))
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        return attention_output, attention_weights

# Initialize Self-Attention
self_attention = SelfAttention(D_MODEL)

# Compute Attention Outputs and Weights
attention_output, attention_weights = self_attention(sequence)

# Visualize Attention Map
plt.figure(figsize=(8, 6))
plt.imshow(attention_weights.detach().numpy(), cmap="viridis")
plt.colorbar()
plt.title("Attention Map")
plt.xlabel("Key Positions")
plt.ylabel("Query Positions")
plt.show()

