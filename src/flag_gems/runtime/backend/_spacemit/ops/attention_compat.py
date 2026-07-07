import torch
from .flash_attention import scaled_dot_product_attention


def _scaled_dot_product_efficient_attention(
    query,
    key,
    value,
    attn_bias=None,
    compute_log_sumexp=False,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
):
    """
    Spacemit backend implementation for _scaled_dot_product_efficient_attention.
    Returns 4-tuple matching aten schema: (output, log_sumexp, philox_seed, philox_offset)
    """
    out = scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_bias,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )

    # Return 4-tuple matching aten::_scaled_dot_product_efficient_attention schema
    # log_sumexp: only compute if requested (currently not implemented, return empty)
    # philox_seed/offset: RNG state for dropout (not used in current impl, return empty)
    log_sumexp = torch.empty(0, dtype=query.dtype, device=query.device)
    philox_seed = torch.empty(0, dtype=torch.int64, device=query.device)
    philox_offset = torch.empty(0, dtype=torch.int64, device=query.device)

    return out, log_sumexp, philox_seed, philox_offset
