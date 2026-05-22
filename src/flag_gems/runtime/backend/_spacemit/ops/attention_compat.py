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
    out = scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_bias,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )
    if compute_log_sumexp:
        return out, None
    return out