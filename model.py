import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
from layers import SublayerSkipConnection,LayerNorm
import copy

def clone(layer, N):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, self_attn, feed_forward, size, dropout):
        super(Encoder, self).__init__()
        self.sub_layers = clone(SublayerSkipConnection(size, dropout), 2)
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.size = size

    def forward(self, x, mask):
        x = self.sub_layers[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sub_layers[1](x, self.feed_forward)

class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.linear = nn.Linear(in_features=d_model,
                                out_features=vocab_size)

    def forward(self, x, exist_mask):
        result, _ = torch.max(self.linear(x), dim=1)
        result = result.masked_fill_(exist_mask == 1, -1e9)
        return F.log_softmax(result, dim=1)
    
class hangmanmodel(nn.Module):
    def __init__(self, encoder: nn.Module, generator, embedding, n_layers: int):
        super(hangmanmodel, self).__init__()
        self.encoder = encoder
        self.layers = clone(encoder, n_layers)
        self.embed = embedding
        self.layer_norm = LayerNorm(encoder.size)
        self.generator = generator

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor):
        x = self.embed(x)
        #print(x.shape)
        for layer in self.layers:    
            x = layer(x, src_mask)
        return self.layer_norm(x)

    @property
    def device(self):
        return self.generator.linear.weight.device
