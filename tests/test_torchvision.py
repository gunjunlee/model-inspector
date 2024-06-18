import pytest

import torch
import torchvision

from model_inspector import ProfilingInterpreter

from .common import REF_FLOPS
from .common import check_flops
from .common import setup_logging


logger = setup_logging(__name__)


def _test_torchvision_model(model_name: str):
    model = getattr(torchvision.models, model_name)(pretrained=False)
    model.eval()
    args = (torch.randn(1, 3, 224, 224), )
    kwargs = {}
    input_example = (args, kwargs)
    interp = ProfilingInterpreter(model, input_example=input_example)
    _ = interp.run(*args, **kwargs)
    table = interp.table
    check_flops(model_name, table.exact_flops.sum(), REF_FLOPS[model_name])
    return table


@pytest.mark.torchvision
@pytest.mark.fast
def test_alexnet():
    _test_torchvision_model("alexnet")


@pytest.mark.torchvision
@pytest.mark.fast
def test_convnext_base():
    _test_torchvision_model("convnext_base")


@pytest.mark.torchvision
def test_convnext_large():
    _test_torchvision_model("convnext_large")


@pytest.mark.torchvision
@pytest.mark.fast
def test_convnext_small():
    _test_torchvision_model("convnext_small")


@pytest.mark.torchvision
@pytest.mark.fast
def test_convnext_tiny():
    _test_torchvision_model("convnext_tiny")


@pytest.mark.torchvision
def test_densenet121():
    _test_torchvision_model("densenet121")


@pytest.mark.torchvision
def test_densenet161():
    _test_torchvision_model("densenet161")


@pytest.mark.torchvision
def test_densenet169():
    _test_torchvision_model("densenet169")


@pytest.mark.torchvision
def test_densenet201():
    _test_torchvision_model("densenet201")


@pytest.mark.torchvision
@pytest.mark.fast
def test_efficientnet_b0():
    _test_torchvision_model("efficientnet_b0")


@pytest.mark.torchvision
@pytest.mark.fast
def test_efficientnet_b1():
    _test_torchvision_model("efficientnet_b1")


@pytest.mark.torchvision
@pytest.mark.fast
def test_efficientnet_b2():
    _test_torchvision_model("efficientnet_b2")


@pytest.mark.torchvision
@pytest.mark.fast
def test_efficientnet_b3():
    _test_torchvision_model("efficientnet_b3")


@pytest.mark.torchvision
def test_efficientnet_b4():
    _test_torchvision_model("efficientnet_b4")


@pytest.mark.torchvision
def test_efficientnet_b5():
    _test_torchvision_model("efficientnet_b5")


@pytest.mark.torchvision
def test_efficientnet_b6():
    _test_torchvision_model("efficientnet_b6")


@pytest.mark.torchvision
def test_efficientnet_b7():
    _test_torchvision_model("efficientnet_b7")


@pytest.mark.torchvision
def test_efficientnet_v2_l():
    _test_torchvision_model("efficientnet_v2_l")


@pytest.mark.torchvision
def test_efficientnet_v2_m():
    _test_torchvision_model("efficientnet_v2_m")


@pytest.mark.torchvision
@pytest.mark.fast
def test_efficientnet_v2_s():
    _test_torchvision_model("efficientnet_v2_s")


@pytest.mark.torchvision
@pytest.mark.fast
def test_googlenet():
    _test_torchvision_model("googlenet")


@pytest.mark.torchvision
@pytest.mark.fast
def test_inception_v3():
    _test_torchvision_model("inception_v3")


@pytest.mark.torchvision
@pytest.mark.fast
def test_mnasnet0_5():
    _test_torchvision_model("mnasnet0_5")


@pytest.mark.torchvision
@pytest.mark.fast
def test_mnasnet0_75():
    _test_torchvision_model("mnasnet0_75")


@pytest.mark.torchvision
@pytest.mark.fast
def test_mnasnet1_0():
    _test_torchvision_model("mnasnet1_0")


@pytest.mark.torchvision
@pytest.mark.fast
def test_mnasnet1_3():
    _test_torchvision_model("mnasnet1_3")


@pytest.mark.torchvision
@pytest.mark.fast
def test_maxvit_t():
    _test_torchvision_model("maxvit_t")


@pytest.mark.torchvision
@pytest.mark.fast
def test_mobilenet_v2():
    _test_torchvision_model("mobilenet_v2")


@pytest.mark.torchvision
@pytest.mark.fast
def test_mobilenet_v3_large():
    _test_torchvision_model("mobilenet_v3_large")


@pytest.mark.torchvision
@pytest.mark.fast
def test_mobilenet_v3_small():
    _test_torchvision_model("mobilenet_v3_small")


@pytest.mark.torchvision
def test_regnet_x_16gf():
    _test_torchvision_model("regnet_x_16gf")


@pytest.mark.torchvision
@pytest.mark.fast
def test_regnet_x_1_6gf():
    _test_torchvision_model("regnet_x_1_6gf")


@pytest.mark.torchvision
def test_regnet_x_32gf():
    _test_torchvision_model("regnet_x_32gf")


@pytest.mark.torchvision
@pytest.mark.fast
def test_regnet_x_3_2gf():
    _test_torchvision_model("regnet_x_3_2gf")


@pytest.mark.torchvision
@pytest.mark.fast
def test_regnet_x_400mf():
    _test_torchvision_model("regnet_x_400mf")


@pytest.mark.torchvision
@pytest.mark.fast
def test_regnet_x_800mf():
    _test_torchvision_model("regnet_x_800mf")


@pytest.mark.torchvision
@pytest.mark.fast
def test_regnet_x_8gf():
    _test_torchvision_model("regnet_x_8gf")


@pytest.mark.torchvision
def test_regnet_y_128gf():
    _test_torchvision_model("regnet_y_128gf")


@pytest.mark.torchvision
def test_regnet_y_16gf():
    _test_torchvision_model("regnet_y_16gf")


@pytest.mark.torchvision
@pytest.mark.fast
def test_regnet_y_1_6gf():
    _test_torchvision_model("regnet_y_1_6gf")


@pytest.mark.torchvision
def test_regnet_y_32gf():
    _test_torchvision_model("regnet_y_32gf")


@pytest.mark.torchvision
@pytest.mark.fast
def test_regnet_y_3_2gf():
    _test_torchvision_model("regnet_y_3_2gf")


@pytest.mark.torchvision
@pytest.mark.fast
def test_regnet_y_400mf():
    _test_torchvision_model("regnet_y_400mf")


@pytest.mark.torchvision
@pytest.mark.fast
def test_regnet_y_800mf():
    _test_torchvision_model("regnet_y_800mf")


@pytest.mark.torchvision
def test_regnet_y_8gf():
    _test_torchvision_model("regnet_y_8gf")


@pytest.mark.torchvision
def test_resnext101_32x8d():
    _test_torchvision_model("resnext101_32x8d")


@pytest.mark.torchvision
def test_resnext101_64x4d():
    _test_torchvision_model("resnext101_64x4d")


@pytest.mark.torchvision
@pytest.mark.fast
def test_resnext50_32x4d():
    _test_torchvision_model("resnext50_32x4d")


@pytest.mark.torchvision
def test_resnet101():
    _test_torchvision_model("resnet101")


@pytest.mark.torchvision
def test_resnet152():
    _test_torchvision_model("resnet152")


@pytest.mark.torchvision
@pytest.mark.fast
def test_resnet18():
    _test_torchvision_model("resnet18")


@pytest.mark.torchvision
@pytest.mark.fast
def test_resnet34():
    _test_torchvision_model("resnet34")


@pytest.mark.torchvision
@pytest.mark.fast
def test_resnet50():
    _test_torchvision_model("resnet50")


@pytest.mark.torchvision
@pytest.mark.fast
def test_shufflenet_v2_x0_5():
    _test_torchvision_model("shufflenet_v2_x0_5")


@pytest.mark.torchvision
@pytest.mark.fast
def test_shufflenet_v2_x1_0():
    _test_torchvision_model("shufflenet_v2_x1_0")


@pytest.mark.torchvision
@pytest.mark.fast
def test_shufflenet_v2_x1_5():
    _test_torchvision_model("shufflenet_v2_x1_5")


@pytest.mark.torchvision
@pytest.mark.fast
def test_shufflenet_v2_x2_0():
    _test_torchvision_model("shufflenet_v2_x2_0")


@pytest.mark.torchvision
@pytest.mark.fast
def test_squeezenet1_0():
    _test_torchvision_model("squeezenet1_0")


@pytest.mark.torchvision
@pytest.mark.fast
def test_squeezenet1_1():
    _test_torchvision_model("squeezenet1_1")


@pytest.mark.torchvision
def test_swin_b():
    _test_torchvision_model("swin_b")


@pytest.mark.torchvision
def test_swin_s():
    _test_torchvision_model("swin_s")


@pytest.mark.torchvision
@pytest.mark.fast
def test_swin_t():
    _test_torchvision_model("swin_t")


@pytest.mark.torchvision
def test_swin_v2_b():
    _test_torchvision_model("swin_v2_b")


@pytest.mark.torchvision
def test_swin_v2_s():
    _test_torchvision_model("swin_v2_s")


@pytest.mark.torchvision
@pytest.mark.fast
def test_swin_v2_t():
    _test_torchvision_model("swin_v2_t")


@pytest.mark.torchvision
@pytest.mark.fast
def test_vgg11_bn():
    _test_torchvision_model("vgg11_bn")


@pytest.mark.torchvision
@pytest.mark.fast
def test_vgg11():
    _test_torchvision_model("vgg11")


@pytest.mark.torchvision
def test_vgg13_bn():
    _test_torchvision_model("vgg13_bn")


@pytest.mark.torchvision
def test_vgg13():
    _test_torchvision_model("vgg13")


@pytest.mark.torchvision
def test_vgg16_bn():
    _test_torchvision_model("vgg16_bn")


@pytest.mark.torchvision
def test_vgg16():
    _test_torchvision_model("vgg16")


@pytest.mark.torchvision
def test_vgg19_bn():
    _test_torchvision_model("vgg19_bn")


@pytest.mark.torchvision
def test_vgg19():
    _test_torchvision_model("vgg19")


@pytest.mark.torchvision
@pytest.mark.fast
def test_vit_b_16():
    _test_torchvision_model("vit_b_16")


@pytest.mark.torchvision
def test_vit_b_32():
    _test_torchvision_model("vit_b_32")


@pytest.mark.torchvision
def test_vit_h_14():
    _test_torchvision_model("vit_h_14")


@pytest.mark.torchvision
def test_vit_l_16():
    _test_torchvision_model("vit_l_16")


@pytest.mark.torchvision
def test_vit_l_32():
    _test_torchvision_model("vit_l_32")


@pytest.mark.torchvision
def test_wide_resnet101_2():
    _test_torchvision_model("wide_resnet101_2")


@pytest.mark.torchvision
@pytest.mark.fast
def test_wide_resnet50_2():
    _test_torchvision_model("wide_resnet50_2")


if __name__ == "__main__":
    table = _test_torchvision_model("efficientnet_b1")
    table.to_csv("torchvision.csv", index=False, sep="\t")
