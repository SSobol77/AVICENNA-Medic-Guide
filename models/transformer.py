#
#  @Siergej Sobolewski
#
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Einsum does matrix multiplication for query*keys for each training example
        # with a different head
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out


'''
### Description:
---
Let's comment on this code step by step, which is an implementation of the attention layer (Self-Attention) in the context 
of transformers using the PyTorch library.

1. **Importing libraries**:
    ```
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    ```
   The necessary components from PyTorch are imported here. `torch` is the main module, `torch.nn` contains classes for 
   creating neural networks, `torch.nn.functional` contains activation functions and other useful utilities.

2. **Definition of the SelfAttention** class:
    ```
    class SelfAttention(nn.Module):
    ```
   The `SelfAttention` class is created, inheriting from `nn.Module`, which is the base class for all neural networks in PyTorch.

3. **Class constructor**:
The constructor of the `SelfAttention' class defines the main parameters and layers:
- `embed_size': the size of the embedding vector.
    - `heads': The number of 'heads' in the attention mechanism that allow the model to focus on different aspects of information.
    - `head_dim`: The size of each head, obtained by dividing `embed_size` by the number of heads.
    - `values`, `keys`, `queries`: linear transformations for values, keys, and queries.
    - `fc_out`: A linear layer to combine the results of all goals.

4. **The `forward` method**:
This method implements the actual attention mechanism:
- `values`, `keys`, `query`: input data for the attention mechanism.
    - `mask`: An optional parameter to "mask" some parts of the input data (for example, to ignore placeholders in data packets).
    - The input data is divided into different "heads".
    - Linear transformations are applied to `values', `keys`, `queries'.
    - The "energy" of attention is calculated using the `torch.einsum` function, which effectively performs operations with tensors.
    - A mask is applied if it is provided.
    - Attention is calculated as softmax from normalized energy.
    - `torch.einsum` is used to get a weighted sum of values.
    - The output data is concatenated and passes through the final linear layer.

This code is one of the key components of the transformer architecture, which allows the model to isolate and process various aspects 
of input data in parallel, which is a significant improvement over previous approaches such as recurrent neural networks.

'''