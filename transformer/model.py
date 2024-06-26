import torch
import torch.nn as nn
import torch.nn.functional as F

import math

# ======================================================================================

# input embeddings for encoder and decoder
class InputEmbedding(nn.Module):
    """Calculates the embedding for input tokens
    """
    def __init__(self, vocab_size: int, d_model: int) -> None:
        """Initialize the input embedding module with a given vocabulary size and model
        dimension

        Args:
            vocab_size: size of the vocabulary. i.e number of tokens present in the 
                vocabulary.
            d_model: dimension to be used for the embedding and the subsequent
                transformer model layers.
        """
        super(InputEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.vocab_size = vocab_size

    def forward(self, x):
        return self.embedding(x)


# positional encoding for encoder and decoder
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(-torch.arange(0, d_model, 2)/d_model * math.log(10000.0))
        pe[:,0::2] = torch.sin(positions * div)
        pe[:,1::2] = torch.cos(positions * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return x


# scaled dot-product attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if attn_mask is not None:
            scores.masked_fill_(attn_mask==0, -1e9)
        attn = F.softmax(scores, dim=-1)
        self.attn = attn
        context = torch.matmul(attn, V)
        return context


# Multi head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.d_h = d_model // n_heads
        self.W_O = nn.Linear(d_model, d_model)
        self.sdpa = ScaledDotProductAttention()

    def forward(self, Q, K, V, attn_mask):
        B, _, C = Q.shape
        assert C == self.d_model, f"Error: the dimension must be {self.d_model}"
        Q = self.W_Q(Q).view(B, Q.size(1), self.n_heads, self.d_h).transpose(1,2)
        K = self.W_K(K).view(B, K.size(1), self.n_heads, self.d_h).transpose(1,2)
        V = self.W_V(V).view(B, V.size(1), self.n_heads, self.d_h).transpose(1,2)
        context_heads = self.sdpa(Q, K, V, attn_mask.unsqueeze(1))
        context = context_heads.transpose(1,2).contiguous().view(B, Q.size(2), self.d_model)
        return self.W_O(context)


# Position-wise feed forward network
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model):
        super(PositionWiseFeedForward, self).__init__()
        self.d_model = d_model
        self.W1 = nn.Linear(d_model, 4 * d_model)
        self.W2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x):
        x = F.relu(self.W1(x))
        return self.W2(x)
    

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha*(x-mean)/(std + self.eps) + self.bias


# Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super(EncoderBlock, self).__init__()
        self.d_model = d_model
        self.mha = MultiHeadAttention(n_heads, d_model)
        self.pos_ffn = PositionWiseFeedForward(d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout1d(dropout)

    def forward(self, x, mask):
        x = self.norm1(x + self.dropout(self.mha(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.pos_ffn(x)))
        return x


# Encoder
class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, dropout):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout1d(dropout)

    def forward(self, x, mask)->torch.Tensor:
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)


# Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super(DecoderBlock, self).__init__()
        self.d_model = d_model
        self.mha1 = MultiHeadAttention(n_heads, d_model)
        self.mha2 = MultiHeadAttention(n_heads, d_model)
        self.pos_ffn = PositionWiseFeedForward(d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout1d(dropout)

    def forward(self, x, x_enc, trg_mask, enc_mask):
        x = self.norm1(x + self.dropout(self.mha1(x, x, x, trg_mask)))
        x = self.norm2(x + self.dropout(self.mha2(x, x_enc, x_enc, enc_mask)))
        x = self.norm3(x + self.dropout(self.pos_ffn(x)))
        return x


# Decoder
class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, dropout):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout1d(dropout)

    def forward(self, x, x_enc, trg_mask, enc_mask):
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, x_enc, trg_mask, enc_mask)
        return self.norm(x)


# Transformer
class Transformer(nn.Module):
    def __init__(
            self, 
            encoder:Encoder, 
            decoder:Decoder, 
            enc_emb:InputEmbedding, 
            dec_emb:InputEmbedding, 
            enc_pos:PositionalEncoding, 
            dec_pos:PositionalEncoding
        ):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.enc_emb = enc_emb
        self.dec_emb = dec_emb
        self.enc_pos = enc_pos
        self.dec_pos = dec_pos 
        self.linear = nn.Linear(decoder.d_model, dec_emb.vocab_size)

    def encode(self, x, mask):
        x = self.enc_emb(x)
        x = self.enc_pos(x)
        x = self.encoder(x, mask)
        return x

    def decode(self, x, x_enc, trg_mask, enc_mask):
        x = self.dec_emb(x)
        x = self.dec_pos(x)
        x = self.decoder(x, x_enc, trg_mask, enc_mask)
        x = self.linear(x)
        return x


def build_transformer(
        n_layers,
        d_model,
        n_heads,
        dropout, 
        src_vocab_size, 
        trg_vocab_size
    ):
    return Transformer(
        Encoder(n_layers, d_model, n_heads, dropout),
        Decoder(d_model, n_heads, n_layers, dropout),
        InputEmbedding(src_vocab_size, d_model),
        InputEmbedding(trg_vocab_size, d_model),
        PositionalEncoding(d_model),
        PositionalEncoding(d_model)
    )
        