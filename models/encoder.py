#
#   @Siergej Sobolewski
#
class Encoder(nn.Module):
    def __init__(
        self, 
        src_vocab_size, 
        embed_size, 
        num_layers, 
        heads, 
        device, 
        forward_expansion, 
        dropout, 
        max_length
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                EncoderLayer(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        out = self.dropout(
            self.word_embedding(x) + self.position_embedding(positions)
        )

        for layer in self.layers:
            out = layer(out, mask)

        return out


'''
### Description:
---
To build a full-fledged encoder in the transformer architecture, we need to stack (stack) several layers of the encoder
that we defined earlier. The full encoder will also include an embedding layer to convert input tokens into vectors, as 
well as positional encoding to add information about the order of tokens in the sequence. 

### Comments on the code:

1. **Initialization**:
- `src_vocab_size`: The size of the dictionary of the source language.
   - `embed_size': The size of the embedding vector.
   - `num_layers': The number of layers of the encoder.
   - `heads`: The number of heads in the attention mechanism.
   - `device': A device for performing calculations (CPU or GPU).
   - `forward_expansion`: The expansion factor for direct communication.
   - `dropout`: The probability of thinning to prevent overfitting.
   - `max_length`: The maximum length of the sequence for positional encoding.

2. **Embedding layer**:
- `word_embedding`: Converts input tokens into vectors.
   - `position_embedding': Adds information about the position of tokens in the sequence.

3. **Encoder Layers**:
   - Created and added to the `ModuleList` for later use.

4. **The `forward` method**:
- `x`: Input data (token indexes).
   - `mask`: A mask to prevent attention to special tokens such as `<pad>'.
   - An embedding layer and positional coding are applied.
   - Consistent application of encoder layers.

This code is a complete transformer encoder that can be used in the transformer architecture for natural language 
processing tasks such as machine translation, text classification, and others.

'''
