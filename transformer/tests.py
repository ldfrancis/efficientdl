import pytest

import torch
import math
import numpy as np

from model import (
    MultiHeadAttention,
    PositionalEncoding, 
    InputEmbedding, 
    ScaledDotProductAttention,
    PositionWiseFeedForward,
    EncoderBlock,
    Encoder,
    DecoderBlock,
    Decoder,
    build_transformer
)

from tokenizer import build_tokenizer
from dataset import EnFrDataset, create_dataloader

import subprocess

# ======================================================================================
# TRANSFORMER
#=======================================================================================

# --------------------------------------------------------------------------------------
# TEST POSITIONAL ENCODING
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "max_len, d_model", [(1, 1), (1, 2), (10,14), (256, 64), (512, 512),]
)
def test_transformer_positional_encoding(max_len: int , d_model: int)->None:
    positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)
    positions = list(range(1,max_len))
    dims = list(range(d_model))
    for pos in positions:
        for dim in dims:
            pos_ = torch.tensor(pos, dtype=torch.float32)
            if dim % 2 == 0: # even
                dim_ = torch.tensor(dim, dtype=torch.float32)
                div = torch.exp(-dim_/d_model * math.log(10000.0))
                pe = torch.sin(pos_ * div).item()
            else: # odd
                dim_ = torch.tensor(dim-1, dtype=torch.float32)
                div = torch.exp(-dim_/d_model * math.log(10000.0))
                pe = torch.cos(pos_ * div).item()
            
            assert pe == positional_encoding.pe[pos, dim].item()


# --------------------------------------------------------------------------------------
# TEST INPUT EMBEDDING
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
        "vocab_size,d_model,inp", [(1000,512,torch.tensor([0,1]))]
)
def test_input_embedding(vocab_size, d_model, inp):
    input_embedding = InputEmbedding(vocab_size=vocab_size, d_model=d_model)
    result = input_embedding(inp)
    assert len(inp.shape) + 1 == len(result.shape)
    assert result.shape[-1] == d_model
           

# --------------------------------------------------------------------------------------
# TEST SCALED DOT PRODUCT ATTENTION
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
        "Q,K,V,attn_mask,result", [(
            torch.tensor([[1,0,1,1],[1,1,1,1]], dtype=torch.float),
            torch.tensor([[1,1,1,1],[0,1,1,0]], dtype=torch.float),
            torch.tensor([[1,0,0,1],[0,1,1,0]], dtype=torch.float),
            None,
            torch.tensor(
                [
                    [0.7311, 0.2689, 0.2689, 0.7311], 
                    [0.7311, 0.2689, 0.2689, 0.7311]
                ]
            )),
            (torch.tensor([[1,0,1,1],[1,1,1,1]], dtype=torch.float),
            torch.tensor([[1,1,1,1],[0,1,1,0]], dtype=torch.float),
            torch.tensor([[1,0,0,1],[0,1,1,0]], dtype=torch.float),
            torch.tensor([[1, 0],[1, 1]], dtype=torch.float),
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 1.0], 
                    [0.7311, 0.2689, 0.2689, 0.7311]
                ]
            ))
        ]
)
def test_attention(Q, K, V, attn_mask, result):
    # ScaledDotProductAttention
    scaledDotProdAttn = ScaledDotProductAttention()
    output = scaledDotProdAttn(Q, K, V, attn_mask)
    np.testing.assert_almost_equal(output.numpy(), result.numpy(), 4)
    # MultiHeadAttention
    mha = MultiHeadAttention(n_heads=2, d_model=4)
    output = mha(Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0), attn_mask)
    assert output.shape == (1,2,4)


# --------------------------------------------------------------------------------------
# TEST POSITIONWISE FEEDFORWARD
# --------------------------------------------------------------------------------------
def test_positionwise_feedforward():
    d_model = 512
    pff = PositionWiseFeedForward(d_model=d_model)
    input_tensor = torch.randn((1, 100, d_model))
    output = pff(input_tensor)
    assert output.shape == (1,100,d_model)


# --------------------------------------------------------------------------------------
# TEST ENCODER BLOCK
# --------------------------------------------------------------------------------------
def test_encoder_block():
    d_model = 512
    n_heads = 8
    dropout = 0.2
    x = torch.randn((1, 100, d_model))
    mask = torch.tril(torch.ones((100,100)), diagonal=0)
    enc = EncoderBlock(d_model, n_heads, dropout)
    output = enc(x, mask)
    for i in range(n_heads):
        assert enc.mha.sdpa.attn[0,i,0,0].item() == 1
        for j in range(100-1):
            for k in range(j+1, 100):
                assert enc.mha.sdpa.attn[0,i,j,k].item() == 0
                assert enc.mha.sdpa.attn[0,i,j,k].item() == 0
    
    assert output.shape == x.shape


# --------------------------------------------------------------------------------------
# TEST ENCODER
# --------------------------------------------------------------------------------------
def test_encoder():
    n_layers = 5
    d_model = 512
    n_heads = 8
    dropout = 0.2
    enc = Encoder(n_layers, d_model, n_heads, dropout)
    x = torch.randn((1, 100, d_model), dtype=torch.float)
    mask = torch.zeros((100,100))
    mask[:, 50] = 1
    output: torch.Tensor = enc(x, mask)
    assert output.shape == x.shape
    for i in range(100):
        for j in range(n_layers):
            for k in range(n_heads):
                assert enc.blocks[j].mha.sdpa.attn[0,k,i,50].item() == 1

    assert output.shape == x.shape


# --------------------------------------------------------------------------------------
# TEST DECODERBLOCK
# --------------------------------------------------------------------------------------
def test_decoderblock():
    d_model = 512
    n_heads = 8
    dropout = 0.2
    n_batch = 8
    src_seqlen = 100
    trg_seqlen = 100

    decoder = DecoderBlock(d_model, n_heads, dropout)

    x_enc = torch.randn((n_batch, src_seqlen, d_model))
    enc_mask = torch.ones((n_batch, trg_seqlen, src_seqlen))
    trg_mask = torch.ones((trg_seqlen, trg_seqlen))
    x = torch.randn((n_batch, trg_seqlen, d_model))
    
    output = decoder(x, x_enc, trg_mask, enc_mask)

    assert output.shape == x.shape


# --------------------------------------------------------------------------------------
# TEST DECODER
# --------------------------------------------------------------------------------------
def test_decoder():
    d_model = 512
    n_heads = 8
    n_layers = 6
    dropout = 0.2
    trg_seqlen = 100
    src_seqlen = 100
    n_batch = 8

    decoder = Decoder(d_model, n_heads, n_layers, dropout)

    x = torch.randn((n_batch, trg_seqlen, d_model))
    x_enc = torch.randn((n_batch, src_seqlen, d_model))
    trg_mask = torch.ones((n_batch, src_seqlen, src_seqlen))
    enc_mask = torch.ones((n_batch, src_seqlen, trg_seqlen))
    output = decoder(x, x_enc, enc_mask, trg_mask)
    assert output.shape == x.shape


# --------------------------------------------------------------------------------------
# TEST TRANSFORMER
# --------------------------------------------------------------------------------------
def test_transformer():
    n_layers = 6
    d_model = 512
    n_heads = 8
    dropout = 0.2
    src_vocab_size = 1000
    trg_vocab_size = 1000

    transformer = build_transformer(
        n_layers, d_model, n_heads, dropout, src_vocab_size, trg_vocab_size
    )

    n_batch = 8
    src_seqlen = 100
    trg_seqlen = 100
    x = torch.randint(1000, (n_batch, src_seqlen))
    x_trg = torch.randint(1000, (n_batch, trg_seqlen))
    src_mask = torch.ones((n_batch, src_seqlen, src_seqlen))
    enc_mask = torch.ones((n_batch, trg_seqlen, src_seqlen))
    trg_mask = torch.ones((n_batch, trg_seqlen, trg_seqlen))

    x_enc = transformer.encode(x, src_mask)
    output = transformer.decode(x_trg, x_enc, trg_mask, enc_mask)

    assert output.shape == (n_batch, trg_seqlen, trg_vocab_size)
    assert x_enc.shape == (n_batch, src_seqlen, d_model)


# ======================================================================================
# TOKENIZER
# ======================================================================================
def test_tokenizer():
    lang1 = "en"
    lang2 = "fr"
    test_folder = "tokenizer-test"
    tokenizer1 = build_tokenizer(lang1, test_folder)
    tokenizer2 = build_tokenizer(lang2, test_folder)
    # breakpoint()
    subprocess.call(f"rm -rf {test_folder}", shell=True)


# ======================================================================================
# DATASET
# ======================================================================================
def test_dataset():
    splits = ["train", "validation", "test"]
    for sp in splits:
        dataset = EnFrDataset(sp)
        dataloader = create_dataloader(sp)
        for batch in dataloader:
            for key in ["src","trg_in","trg_out","trg_mask","src_mask","trg_src_mask"]:
                assert key in batch
            
    