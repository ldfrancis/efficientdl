import pytest

import torch
import math
import numpy as np

from transformer import PositionalEncoding, InputEmbedding, ScaledDotProductAttention

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
            ))
        ]
)
def test_scaled_dot_product_attention(Q, K, V, attn_mask, result):
    scaledDotProdAttn = ScaledDotProductAttention()
    output = scaledDotProdAttn(Q, K, V, attn_mask)
    np.testing.assert_almost_equal(output.numpy(), result.numpy(), 4)

