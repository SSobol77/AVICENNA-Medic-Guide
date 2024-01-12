#
# @Siergej Sobolewski
#
class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderLayer(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

        return out

'''
### Description:
---
This is the code of a full-fledged decoder in the transformer architecture. This decoder uses previously defined 
decoder layers (`DecoderLayer') and interacts with data from the encoder.

### In this code:

1. **Initialization**:
- `trg_vocab_size`: The size of the target language dictionary.
- `embed_size': The size of the attachment.
- `num_layers': The number of decoder layers.
- `heads`: The number of heads in the attention mechanism.
- `forward_expansion`: The expansion factor for a straight connected layer.
- `dropout': The probability of disabling neurons.
- `device': Device (CPU or GPU).
- `max_length`: The maximum length of the sequence.

2. **Embedding layer**:
- Converting input tokens into vectors.

3. **Positional coding**:
- Adding information about the position of tokens in the sequence.

4. **Decoder Layers**:
- Application of several decoder layers, each of which uses attention and a direct connected layer.

5. **The `forward` method**:
- Processing of input data, positional encoding and interaction with the output of the encoder.
- `enc_out': Output of the encoder.
- `src_mask` and `trg_mask': Masks for source and target data.

6. **Output linear layer**:
- Conversion of vectors into probabilities of the next token.

This decoder is a key component of the transformer architecture for tasks requiring sequence generation, for example, in machine translation.

'''
