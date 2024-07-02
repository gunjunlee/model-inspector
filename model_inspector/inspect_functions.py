import torch

import logging


logger = logging.getLogger(__name__)


# https://github.com/pytorch/pytorch/blob/b0282071c48860fcf8f4c1025bc207138173617b/aten/src/ATen/native/Convolution.cpp#L1168
# at::Tensor convolution(
#     const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt,
#     IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
#     bool transposed, IntArrayRef output_padding, int64_t groups)
def _calc_flops_conv(*args, **kwargs):
    assert len(args) == 9, f"Expected 9 arguments, got {len(args)}"
    input, weight, bias_opt, stride, padding, dilation, transposed, output_padding, groups = args

    assert transposed is False, f"Transposed convolutions not supported"
    assert input.shape[1] == weight.shape[1] * groups, f"Input channels should be equal to weight channels * groups"

    b, ic, h, w = input.shape
    oc, ic, kh, kw = weight.shape

    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    oh = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    ow = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    flops = 2 * b * oh * ow * kh * kw * ic * oc
    if bias_opt is not None:
        flops += b * oh * ow * oc

    return flops

# https://github.com/pytorch/pytorch/blob/b0282071c48860fcf8f4c1025bc207138173617b/aten/src/ATen/native/transformers/attention.cpp#L638
def _calc_flops_attention(*args, **kwargs):
    assert len(args) == 3, "Scaled dot product attention should have 3 arguments"
    assert all(isinstance(arg, torch.Tensor) for arg in args), "All arguments should be tensors"
    q, k, v = args
    q = q.reshape(-1, q.shape[-2], q.shape[-1])
    k = k.reshape(-1, k.shape[-2], k.shape[-1])
    v = v.reshape(-1, v.shape[-2], v.shape[-1])
    assert q.shape[0] == k.shape[0] == v.shape[0], "Batch size should be the same for query, key and value"
    assert k.shape[1] == v.shape[1], "Number of heads should be the same for key and value"
    assert q.shape[2] == k.shape[2], "Query and key dimension should be the same"
    batch, trg_seq_len, dim = q.shape
    batch, src_seq_len, dim = k.shape
    batch, src_seq_len, dimv = v.shape

    flops = 2 * batch * trg_seq_len * src_seq_len * dim  # q @ k
    flops += batch * trg_seq_len * src_seq_len  # scaling
    flops += 2 * batch * trg_seq_len * src_seq_len * dimv  # softmax(q @ k) @ v

    return flops


# https://github.com/pytorch/pytorch/blob/34e94c507ab508ab6dea61373dc93450a9618db8/aten/src/ATen/native/LinearAlgebra.cpp#L196
# TORCH_META_FUNC(mm)(const Tensor& self, const Tensor& mat2) {
def _calc_flops_mm(*args, **kwargs):
    assert len(args) == 2, "Expected 2 arguments"
    assert len(kwargs) == 0, "Expected 0 keyword arguments"
    assert all(isinstance(arg, torch.Tensor) for arg in args), "All arguments should be tensors"
    input, mat2 = args
    if len(input.shape) != 2:
        logger.warning(f"Matrix multiplication with input shape {input.shape} is not tested")
    if len(mat2.shape) != 2:
        logger.warning(f"Matrix multiplication with mat2 shape {mat2.shape} is not tested")
    assert input.shape[-1] == mat2.shape[0], "Matrix multiplication dimensions do not match"
    mul_shape = input.shape + mat2.shape[1:]
    flops = 2 * mul_shape.numel()
    return flops


# https://github.com/pytorch/pytorch/blob/34e94c507ab508ab6dea61373dc93450a9618db8/aten/src/ATen/native/LinearAlgebra.cpp#L188
# TORCH_META_FUNC(addmm)(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha) {
def _calc_flops_addmm(*args, **kwargs):
    assert len(args) == 3, "Expected 3 arguments"
    assert len(kwargs) == 0, "Expected 0 keyword arguments"
    assert all(isinstance(arg, torch.Tensor) for arg in args), "All arguments should be tensors"
    input, mat1, mat2 = args
    if len(input.shape) != 1:
        logger.warning(f"Addmm operation with input shape {input.shape} is not tested")
    if len(mat1.shape) != 2:
        logger.warning(f"Addmm operation with mat1 shape {mat1.shape} is not tested")
    if len(mat2.shape) != 2:
        logger.warning(f"Addmm operation with mat2 shape {mat2.shape} is not tested")
    assert mat1.shape[-1] == mat2.shape[0], "Matrix multiplication dimensions do not match"
    mul_shape = mat1.shape + mat2.shape[1:]
    flops = 2 * mul_shape.numel()
    return flops


def _calc_flops_mul(*args, **kwargs):
    assert len(args) == 2, "Expected 2 arguments"
    assert len(kwargs) == 0, "Expected 0 keyword arguments"
    assert all(isinstance(arg, torch.Tensor) for arg in args), "All arguments should be tensors"
    input, mat2 = args
    output_shape = torch.broadcast_shapes(input.shape, mat2.shape)
    return 0


def _calc_flops_add(*args, **kwargs):
    assert len(args) == 2, "Expected 2 arguments"
    assert len(kwargs) == 0, "Expected 0 keyword arguments"
    assert all(isinstance(arg, torch.Tensor) for arg in args), "All arguments should be tensors"
    input, mat2 = args
    output_shape = torch.broadcast_shapes(input.shape, mat2.shape)
    return 0


# https://github.com/pytorch/pytorch/blob/34e94c507ab508ab6dea61373dc93450a9618db8/aten/src/ATen/native/LinearAlgebra.cpp#L329
# TORCH_META_FUNC(bmm)(const Tensor& self, const Tensor& mat2) {
def _calc_flops_bmm(*args, **kwargs):
    assert len(args) == 2, "Expected 2 arguments"
    assert len(kwargs) == 0, "Expected 0 keyword arguments"
    assert all(isinstance(arg, torch.Tensor) for arg in args), "All arguments should be tensors"
    input, mat2 = args
    assert len(input.shape) == 3, "Input should be a 3D tensor"
    assert len(mat2.shape) == 3, "mat2 should be a 3D tensor"
    assert input.shape[0] == mat2.shape[0], "Batch size should be the same for input and mat2"
    assert input.shape[2] == mat2.shape[1], "Inner dimensions should match for input and mat2"
    mul_shape = input.shape + mat2.shape[2:]
    flops = 2 * mul_shape.numel()
    return flops


# https://github.com/pytorch/pytorch/blob/34e94c507ab508ab6dea61373dc93450a9618db8/aten/src/ATen/native/LinearAlgebra.cpp#L333
# TORCH_META_FUNC(baddbmm)(const Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
def _calc_flops_baddbmm(*args, **kwargs):
    raise NotImplementedError("baddbmm not implemented")


# https://github.com/pytorch/pytorch/blob/b0282071c48860fcf8f4c1025bc207138173617b/torch/csrc/profiler/util.cpp#L405
FLOPABLE_OPS = {
    # manual
    "aten::_scaled_dot_product_flash_attention": _calc_flops_attention,
    "aten::convolution": _calc_flops_conv,

    # auto
    "aten::conv2d": _calc_flops_conv,
    "aten::mm": _calc_flops_mm,
    "aten::addmm": _calc_flops_addmm,
    "aten::mul": _calc_flops_mul,
    "aten::add": _calc_flops_add,
    "aten::bmm": _calc_flops_bmm,
    "aten::baddbmm": _calc_flops_baddbmm,
}

# aten::add.Tensor
# aten::mul.Tensor
# aten::div.Tensor


def get_flops_counter(op_name):
    if op_name in FLOPABLE_OPS:
        return FLOPABLE_OPS[op_name]
    else:
        raise NotImplementedError(f"Operation {op_name} not supported")
