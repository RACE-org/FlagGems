import torch


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
    Stub implementation for _scaled_dot_product_efficient_attention.
    Returns empty tensors matching the aten schema:
    (output, log_sumexp, philox_seed, philox_offset)

    Vendor-specific backends (e.g., spacemit) override this via extend_op registration.
    """
    # Return 4-tuple matching aten::_scaled_dot_product_efficient_attention schema
    output = torch.empty_like(query)
    log_sumexp = torch.empty(0, dtype=query.dtype, device=query.device)
    philox_seed = torch.empty(0, dtype=torch.int64, device=query.device)
    philox_offset = torch.empty(0, dtype=torch.int64, device=query.device)
    return output, log_sumexp, philox_seed, philox_offset
