import pytest
import torch
import os,sys,time

import flag_gems
from .accuracy_utils import gems_assert_close, to_reference, gems_assert_cosine_similarity
from . import accuracy_utils as utils


@pytest.mark.gelu
@pytest.mark.parametrize("shape", [(1,4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("approximate", ["none","tanh"])
def test_accuracy_gelu(shape, dtype, approximate):
    res_inp = torch.randn(shape, dtype=dtype)
    #该输入时, gelu输出与输入一致
    res_inp[0,:] = 10.1875
    res_inp = res_inp.to(device=flag_gems.device)
    ref_inp = to_reference(res_inp, True)
    #print('ref_inp:', ref_inp)
    ref_out = torch.nn.functional.gelu(ref_inp, approximate=approximate)
    #print('ref_out:', ref_out)
    #print('res_inp:', res_inp.cpu())
    with flag_gems.use_gems():
        res_out = torch.nn.functional.gelu(res_inp, approximate=approximate)
    #print('res_out:', res_out.cpu())
    atol = 1e-4
    gems_assert_close(res_out, ref_out, dtype, atol=atol)


@pytest.mark.tanh
@pytest.mark.parametrize("shape", [(1,4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_accuracy_tanh(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype)
    #输入大于44.3时硬件输出nan
    res_inp[0,:] = 88
    res_inp = res_inp.to(device=flag_gems.device)
    ref_inp = to_reference(res_inp, False)
    #print('\nref_inp dtype:', ref_inp.dtype)
    #print('ref_inp:', ref_inp)
    ref_out = torch.tanh(ref_inp)
    #print('ref_out:', ref_out)
    #print('res_inp dtype:', res_inp.dtype)
    #print('res_inp:', res_inp.cpu())
    with flag_gems.use_gems():
        res_out = torch.tanh(res_inp)
    #print('res_out dtype:', res_out.dtype)
    #print('res_out:', res_out.cpu())
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.mm
@pytest.mark.parametrize("M, N, K", [(256, 256, 256)])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_accuracy_mm_transB(M, N, K, dtype):
    mat1 = torch.randn((M, K), dtype=dtype).to(device=flag_gems.device)
    #B矩阵行主序存储
    mat2 = torch.randn((K, N), dtype=dtype).to(device=flag_gems.device)
    #B矩阵列主序存储,与行主序时性能应该相当
    mat2col = torch.randn((N, K), dtype=dtype).to(device=flag_gems.device).t()
    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)
    ref_mat2col = to_reference(mat2col, True)

    ref_out = torch.mm(ref_mat1, ref_mat2)
    ref_out_col = torch.mm(ref_mat1, ref_mat2col)
    with flag_gems.use_gems():
        # warmup
        res_out = torch.mm(mat1, mat2)
        res_out_col = torch.mm(mat1, mat2col)
        t1 = time.time()
        res_out = torch.mm(mat1, mat2)
        t2 = time.time()
        res_out_col = torch.mm(mat1, mat2col)
        t3 = time.time()

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)
    gems_assert_close(res_out_col, ref_out_col, dtype, reduce_dim=K)
    cost_ration = (t2-t1)/(t3-t2)
    print(f"B矩阵行主序存储e2e耗时: {(t2-t1)*1000*1000}us")
    print(f"B矩阵列主序存储e2e耗时: {(t3-t2)*1000*1000}us")
    assert cost_ration < 2.0 and cost_ration > 0.5


@pytest.mark.mm
@pytest.mark.parametrize("M, N, K", [(10, 151680, 5120)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_mm_big_shape(M, N, K, dtype):
    x = torch.randn((M, K), dtype=dtype, requires_grad=True).to('txda')
    w = torch.randn((K, N), dtype=dtype, requires_grad=True).to('txda')
    #print("\nx.shape:", x.shape)
    with flag_gems.use_gems():
        logits = torch.mm(x, w)
        xt = x.transpose(0, 1).contiguous()
        grad_w = torch.mm(xt, logits)
        wt = w.transpose(0, 1).contiguous()
        grad_x = torch.mm(logits, wt)
        
        ref_logits = to_reference(logits, True)
        ref_wt = to_reference(wt, True)
        ref_grad_x = torch.mm(ref_logits, ref_wt)

    if dtype == torch.float32:
        gems_assert_cosine_similarity(grad_x, ref_grad_x, dtype)
    else:
        gems_assert_close(grad_x, ref_grad_x, dtype, reduce_dim=K)




@pytest.mark.logical_and_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype",
    utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES,
)
def test_logical_and_(shape, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    if dtype in utils.ALL_FLOAT_DTYPES:
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    elif dtype in utils.ALL_INT_DTYPES:
        inp1 = torch.randint(-1000, 1000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
        inp2 = torch.randint(-1000, 1000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    elif dtype in utils.BOOL_TYPES:
        inp1 = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
        inp2 = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )

    ref_inp1 = utils.to_reference(inp1.clone())
    ref_inp2 = utils.to_reference(inp2)

    ref_out = ref_inp1.logical_and_(ref_inp2)
    with flag_gems.use_gems():
        res_out = inp1.logical_and_(inp2) #触发mk-to-tx81  int32tobf16场景

    utils.gems_assert_equal(res_out, ref_out)




