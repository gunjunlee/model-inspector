import torch

import logging
import operator

logger = logging.getLogger(__name__)


def _inspect_function_add(profiler, node):
    assert node.op == "call_function"
    args, kwargs = profiler.fetch_args_kwargs_from_env(node)
    assert len(args) == 2, "Addition should have 2 arguments"
    num_ops = 0
    if any(isinstance(arg, torch.Tensor) for arg in args):
        x, y = args
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)
        num_ops = torch.broadcast_shapes(x.shape, y.shape).numel()
    node_name = list(node.meta.get("nn_module_stack", [""]))[-1]
    return ProfileResult(node, node_name, "Add", ops=num_ops)


def _inspect_function_mul(profiler, node):
    assert node.op == "call_function"
    args, kwargs = profiler.fetch_args_kwargs_from_env(node)
    assert len(args) == 2, "Multiplication should have 2 arguments"
    num_ops = 0
    if any(isinstance(arg, torch.Tensor) for arg in args):
        x, y = args
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)
        num_ops = torch.broadcast_shapes(x.shape, y.shape).numel()
    node_name = list(node.meta.get("nn_module_stack", [""]))[-1]
    return ProfileResult(node, node_name, "Mul", ops=num_ops)


def _inspect_function_addmm(profiler, node):
    assert node.op == "call_function"
    args, kwargs = profiler.fetch_args_kwargs_from_env(node)
    assert len(args) == 3, "addmm should have 3 arguments"
    assert all(isinstance(arg, torch.Tensor) for arg in args), "All arguments should be tensors"
    x, y, z = args
    y = y.view(-1, y.shape[-1])
    z = z.view(z.shape[0], -1)
    assert y.shape[-1] == z.shape[0], "Inner dimensions should be the same"
    num_mads = y.shape[0] * y.shape[1] * z.shape[1]
    node_name = list(node.meta.get("nn_module_stack", [""]))[-1]
    return ProfileResult(node, node_name, "Addmm", mads=num_mads)


def _inspect_function_scale_dot_product_attention(profiler, node):
    assert node.op == "call_function"
    args, kwargs = profiler.fetch_args_kwargs_from_env(node)
    assert len(args) == 3, "Scaled dot product attention should have 3 arguments"
    assert all(isinstance(arg, torch.Tensor) for arg in args), "All arguments should be tensors"
    q, k, v = args
    q = q.view(-1, q.shape[-2], q.shape[-1])
    k = k.view(-1, k.shape[-2], k.shape[-1])
    v = v.view(-1, v.shape[-2], v.shape[-1])
    assert q.shape[0] == k.shape[0] == v.shape[0], "Batch size should be the same for query, key and value"
    assert k.shape[1] == v.shape[1], "Number of heads should be the same for key and value"
    assert q.shape[2] == k.shape[2], "Query and key dimension should be the same"
    batch, trg_seq_len, dim = q.shape
    batch, src_seq_len, dim = k.shape
    batch, src_seq_len, dimv = v.shape

    mads = batch * trg_seq_len * src_seq_len * dim
    ops = batch * trg_seq_len * src_seq_len
    mads += batch * trg_seq_len * src_seq_len * dimv

    node_name = list(node.meta["nn_module_stack"])[-1]
    return ProfileResult(node, node_name, "Attention", mads=mads, ops=ops)


def _inspect_module_conv2d(profiler, node):
    assert node.op == "call_module"
    module = profiler.fetch_attr(node.target)
    assert isinstance(module, torch.nn.Conv2d)
    args, kwargs = profiler.fetch_args_kwargs_from_env(node)
    assert len(args) == 1, "Conv2d should have 1 argument"
    assert isinstance(args[0], torch.Tensor), "Argument should be a tensor"
    b, c, h, w = args[0].shape
    kh, kw = module.kernel_size
    ic, oc = module.in_channels, module.out_channels
    sh, sw = module.stride
    ph, pw = module.padding
    dh, dw = module.dilation
    oh = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    ow = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    mads = b * oh * ow * kh * kw * ic * oc
    ops = 0
    if module.bias is not None:
        ops += b * oh * ow * oc
    node_name = list(node.meta["nn_module_stack"])[-1]
    return ProfileResult(node, node_name, "Conv2d", mads=mads, ops=ops)


def _inspect_module_linear(profiler, node):
    assert node.op == "call_module"
    module = profiler.fetch_attr(node.target)
    assert isinstance(module, torch.nn.Linear)
    args, kwargs = profiler.fetch_args_kwargs_from_env(node)
    assert len(args) == 1, "Linear should have 1 argument"
    assert isinstance(args[0], torch.Tensor), "Argument should be a tensor"
    inp = args[0]
    inp = inp.view(-1, inp.shape[-1])
    b, c = inp.shape
    mads = b * c * module.out_features
    ops = 0
    if module.bias is not None:
        ops += b * module.out_features
    node_name = list(node.meta["nn_module_stack"])[-1]
    return ProfileResult(node, node_name, "Linear", mads=mads, ops=ops)

_INSPECTABLE_FUNCTIONS = {
    # https://github.com/pytorch/pytorch/blob/main/torch/_decomp/decompositions.py
    torch._decomp.decompositions._addmm_activation: None,
    torch._decomp.decompositions._euclidean_dist: None,
    torch._decomp.decompositions.addmm: None,
    torch._decomp.decompositions.addmv: None,
    torch._decomp.decompositions.baddbmm: None,
    torch._decomp.decompositions.matmul: None,
    torch._decomp.decompositions.mv: None,

    # https://github.com/pytorch/pytorch/blob/main/torch/_refs/__init__.py


    # https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py
    torch._C._nn._conv_depthwise2d: None,
    torch._C._nn.conv_depthwise3d: None,
    torch._C._nn.linear: None,
    torch._C._nn.mkldnn_linear: None,
    torch._C._nn.scaled_dot_product_attention: None,
    torch._C._nn.slow_conv3d: None,
    torch._C._nn.slow_conv_dilated2d: None,
    torch._C._nn.slow_conv_dilated3d: None,
    torch._C._nn.slow_conv_transpose2d: None,
    torch._C._nn.slow_conv_transpose3d: None,
    torch._C._nn.thnn_conv2d: None,
}

# torch._decomp.decompositions
_INSPECTABLE_FUNCTIONS_NONE = ["_fused_dropout_decomposition", "_log_softmax", "_log_softmax_backward_data", "_reflection_pad", "_replication_pad", "_reshape_alias", "_softmax", "_softmax_backward_data", "_to_copy", "adaptive_avg_pool2d", "affine_grid_generator", "aminmax", "arange_default", "binary_cross_entropy", "binary_cross_entropy_backward", "binary_cross_entropy_with_logits", "col2im", "cudnn_batch_norm", "cudnn_batch_norm_backward", "diagonal_backward", "dist", "elu_backward", "embedding", "embedding_dense_backward", "floor_divide", "gelu_backward", "glu_backward", "grid_sampler_2d", "hardsigmoid", "hardsigmoid_backward", "hardswish", "hardswish_backward", "hardtanh_backward", "im2col", "index_add", "index_copy", "isin", "leaky_relu_backward", "log_sigmoid_backward", "log_sigmoid_forward", "mse_loss", "mse_loss_backward", "multi_margin_loss", "multilabel_margin_loss_forward", "nansum", "native_batch_norm", "native_dropout", "native_dropout_backward", "nll_loss2d_backward", "nll_loss2d_forward", "nll_loss_backward", "nll_loss_forward", "nop_decomposition", "rrelu_with_noise", "rrelu_with_noise_backward", "select_backward", "sigmoid_backward", "silu", "silu_backward", "slice_backward", "smooth_l1_loss", "soft_margin_loss", "soft_margin_loss_backward", "softplus_backward", "take", "tanh_backward", "threshold_backward", "unfold_backward", "uniform", "upsample_bicubic2d_vec", "upsample_bilinear2d", "upsample_linear1d", "upsample_trilinear3d"]
_INSPECTABLE_FUNCTIONS.update(
    {getattr(torch._decomp.decompositions, f): None for f in _INSPECTABLE_FUNCTIONS_NONE}
)

# torch._refs
_INSPECTABLE_FUNCTIONS_NONE = ["_block_diag_iterable", "abs", "abs_", "acos", "acos_", "acosh", "acosh_", "add", "add_", "addcdiv", "addcdiv_", "addcmul", "addcmul_", "addr", "all", "any", "arange", "as_strided_scatter", "asin", "asin_", "asinh", "asinh_", "atan", "atan2", "atan2_", "atan_", "atanh", "atanh_", "bitwise_and", "bitwise_and_", "bitwise_left_shift", "bitwise_left_shift_", "bitwise_not", "bitwise_not_", "bitwise_or", "bitwise_or_", "bitwise_right_shift", "bitwise_right_shift_", "bitwise_xor", "bitwise_xor_", "bucketize", "cat", "cauchy", "cauchy_", "ceil", "ceil_", "clamp", "clamp_", "clamp_max", "clamp_max_", "clamp_min", "clamp_min_", "clone", "column_stack", "conj_physical", "conj_physical_", "constant_pad_nd", "copysign", "copysign_", "cos", "cos_", "cosh", "cosh_", "count_nonzero", "deg2rad", "deg2rad_", "diag", "diag_embed", "diagonal_scatter", "digamma", "digamma_", "div", "div_", "dot", "dstack", "empty", "empty_like", "empty_permuted", "empty_strided", "eq", "eq_", "erf", "erf_", "erfc", "erfc_", "erfinv", "erfinv_", "exp", "exp2", "exp2_", "exp_", "expm1", "expm1_", "exponential", "exponential_", "eye", "fill", "flip", "float_power", "float_power_", "floor", "floor_", "floor_divide", "floor_divide_", "fmax", "fmin", "fmod", "fmod_", "frac", "frac_", "frexp", "full", "gcd", "gcd_", "ge", "ge_", "geometric", "geometric_", "gt", "gt_", "heaviside", "heaviside_", "hstack", "hypot", "hypot_", "i0", "i0_", "igamma", "igamma_", "igammac", "igammac_", "index_add", "index_copy", "index_fill", "index_select", "isfinite", "isinf", "isnan", "isneginf", "isposinf", "isreal", "lcm", "lcm_", "le", "le_", "lerp", "lerp_", "lgamma", "lgamma_", "linspace", "log", "log10", "log10_", "log1p", "log1p_", "log2", "log2_", "log_", "log_normal", "log_normal_", "log_softmax", "logaddexp", "logaddexp2", "logical_and", "logical_and_", "logical_not", "logical_not_", "logical_or", "logical_or_", "logical_xor", "logical_xor_", "logspace", "logsumexp", "lt", "lt_", "masked_fill", "maximum", "minimum", "mul", "mul_", "nan_to_num", "nan_to_num_", "native_layer_norm", "ne", "ne_", "neg", "neg_", "new_empty", "new_empty_strided", "new_full", "new_ones", "new_zeros", "nextafter", "nextafter_", "norm", "normal", "ones", "ones_like", "pow", "pow_", "rad2deg", "rad2deg_", "randn", "reciprocal", "reciprocal_", "remainder", "remainder_", "renorm", "repeat", "roll", "rot90", "round", "rsqrt", "rsqrt_", "rsub", "sgn", "sgn_", "sigmoid", "sigmoid_", "sign", "sign_", "signbit", "sin", "sin_", "sinc", "sinc_", "sinh", "sinh_", "softmax", "sqrt", "sqrt_", "square", "square_", "stack", "std", "std_mean", "sub", "sub_", "take_along_dim", "tan", "tan_", "tanh", "tanh_", "trace", "tril", "tril_", "tril_indices", "triu", "triu_", "triu_indices", "true_divide", "true_divide_", "trunc", "trunc_", "trunc_divide", "unfold_copy", "var", "var_mean", "vdot", "vstack", "where", "xlogy", "xlogy_", "zero", "zero_", "zeros", "zeros_like"]
_INSPECTABLE_FUNCTIONS.update(
    {getattr(torch._refs, f): None for f in _INSPECTABLE_FUNCTIONS_NONE}
)

def _check_module(module, filter=lambda _: True):
    _functions = list(
        p for p in dir(module)
        if (
            not p.startswith("__")
            and not p[0].isupper()
            and callable(getattr(module, p))
            and filter(getattr(module, p))
        )
    )
    for _func_name in sorted(_functions):
        _func = getattr(module, _func_name)
        if _func not in _INSPECTABLE_FUNCTIONS:
            logger.warning(f"module function {module.__name__}.{_func_name} is not supported yet")

_check_module(torch._decomp.decompositions, lambda x: hasattr(x, "_torch_decompositions_out_wrapper"))
_check_module(torch._refs, lambda x: hasattr(x, "_torch_decompositions_out_wrapper"))
_check_module(torch._C._nn)
_check_module(torch)
_check_module(operator)













module function torch._C._nn._pad_circular is not supported yet
module function torch._C._nn._pad_enum is not supported yet
module function torch._C._nn._parse_to is not supported yet
module function torch._C._nn._test_ambiguous_defaults is not supported yet
module function torch._C._nn._test_optional_filled_intlist is not supported yet
module function torch._C._nn._test_optional_floatlist is not supported yet
module function torch._C._nn._test_optional_intlist is not supported yet
module function torch._C._nn._test_string_default is not supported yet
module function torch._C._nn._test_warn_in_autograd is not supported yet
module function torch._C._nn._upsample_bicubic2d_aa is not supported yet
module function torch._C._nn._upsample_bilinear2d_aa is not supported yet
module function torch._C._nn._upsample_nearest_exact1d is not supported yet
module function torch._C._nn._upsample_nearest_exact2d is not supported yet
module function torch._C._nn._upsample_nearest_exact3d is not supported yet
module function torch._C._nn.adaptive_avg_pool2d is not supported yet
module function torch._C._nn.adaptive_avg_pool3d is not supported yet
module function torch._C._nn.adaptive_max_pool2d is not supported yet
module function torch._C._nn.adaptive_max_pool3d is not supported yet
module function torch._C._nn.avg_pool2d is not supported yet
module function torch._C._nn.avg_pool3d is not supported yet
module function torch._C._nn.binary_cross_entropy is not supported yet
module function torch._C._nn.col2im is not supported yet
module function torch._C._nn.cross_entropy_loss is not supported yet
module function torch._C._nn.elu is not supported yet
module function torch._C._nn.elu_ is not supported yet
module function torch._C._nn.flatten_dense_tensors is not supported yet
module function torch._C._nn.fractional_max_pool2d is not supported yet
module function torch._C._nn.fractional_max_pool3d is not supported yet
module function torch._C._nn.gelu is not supported yet
module function torch._C._nn.gelu_ is not supported yet
module function torch._C._nn.glu is not supported yet
module function torch._C._nn.hardsigmoid is not supported yet
module function torch._C._nn.hardsigmoid_ is not supported yet
module function torch._C._nn.hardswish is not supported yet
module function torch._C._nn.hardswish_ is not supported yet
module function torch._C._nn.hardtanh is not supported yet
module function torch._C._nn.hardtanh_ is not supported yet
module function torch._C._nn.huber_loss is not supported yet
module function torch._C._nn.im2col is not supported yet
module function torch._C._nn.l1_loss is not supported yet
module function torch._C._nn.leaky_relu is not supported yet
module function torch._C._nn.leaky_relu_ is not supported yet
module function torch._C._nn.log_sigmoid is not supported yet
module function torch._C._nn.max_pool2d_with_indices is not supported yet
module function torch._C._nn.max_pool3d_with_indices is not supported yet
module function torch._C._nn.max_unpool2d is not supported yet
module function torch._C._nn.max_unpool3d is not supported yet
module function torch._C._nn.mish is not supported yet
module function torch._C._nn.mish_ is not supported yet
module function torch._C._nn.mkldnn_reorder_conv2d_weight is not supported yet
module function torch._C._nn.mkldnn_reorder_conv3d_weight is not supported yet
module function torch._C._nn.mse_loss is not supported yet
module function torch._C._nn.multi_margin_loss is not supported yet
module function torch._C._nn.multilabel_margin_loss is not supported yet
module function torch._C._nn.nll_loss is not supported yet
module function torch._C._nn.nll_loss2d is not supported yet
module function torch._C._nn.nll_loss_nd is not supported yet
module function torch._C._nn.one_hot is not supported yet
module function torch._C._nn.pad is not supported yet
module function torch._C._nn.pad_sequence is not supported yet
module function torch._C._nn.reflection_pad1d is not supported yet
module function torch._C._nn.reflection_pad2d is not supported yet
module function torch._C._nn.reflection_pad3d is not supported yet
module function torch._C._nn.relu6 is not supported yet
module function torch._C._nn.relu6_ is not supported yet
module function torch._C._nn.replication_pad1d is not supported yet
module function torch._C._nn.replication_pad2d is not supported yet
module function torch._C._nn.replication_pad3d is not supported yet
module function torch._C._nn.rrelu_with_noise is not supported yet
module function torch._C._nn.rrelu_with_noise_ is not supported yet
module function torch._C._nn.silu is not supported yet
module function torch._C._nn.silu_ is not supported yet
module function torch._C._nn.smooth_l1_loss is not supported yet
module function torch._C._nn.soft_margin_loss is not supported yet
module function torch._C._nn.softplus is not supported yet
module function torch._C._nn.softshrink is not supported yet
module function torch._C._nn.unflatten_dense_tensors is not supported yet
module function torch._C._nn.upsample_bicubic2d is not supported yet
module function torch._C._nn.upsample_bilinear2d is not supported yet
module function torch._C._nn.upsample_linear1d is not supported yet
module function torch._C._nn.upsample_nearest1d is not supported yet
module function torch._C._nn.upsample_nearest2d is not supported yet
module function torch._C._nn.upsample_nearest3d is not supported yet
module function torch._C._nn.upsample_trilinear3d is not supported yet