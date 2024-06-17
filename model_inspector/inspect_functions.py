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

    assert groups == 1, f"Groups > 1 not supported"
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
