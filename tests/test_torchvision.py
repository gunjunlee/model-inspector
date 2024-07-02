import pytest

import torch
import torchvision

from model_inspector import ProfilingInterpreter

from .common import check_flops
from .common import setup_logging
from .common import get_vision_spec
from .common import get_available_devices


DEBUG = False


logger = setup_logging(__name__)


TEST_DEVICES = get_available_devices()


def _test_torchvision_model(model_name: str, device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[{model_name}] device: {device}")

    ref_flops, input_size = get_vision_spec(model_name)
    model = getattr(torchvision.models, model_name)(pretrained=False)
    model = model.eval().to(device)
    args = (torch.randn(1, 3, input_size, input_size).to(device), )
    kwargs = {}
    input_example = (args, kwargs)
    interp = ProfilingInterpreter(model, input_example=input_example)
    _ = interp.run(*args, **kwargs)
    if not DEBUG:
        test_name = f"{model_name}_{device}"
        check_flops(test_name, interp.flops, ref_flops)
        if device == "cuda":
            check_flops(test_name, interp.cuda_flops, ref_flops)
    return interp


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_alexnet(device):
    _test_torchvision_model("alexnet", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_convnext_base(device):
    _test_torchvision_model("convnext_base", device=device)


@pytest.mark.torchvision
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_convnext_large(device):
    _test_torchvision_model("convnext_large", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_convnext_small(device):
    _test_torchvision_model("convnext_small", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_convnext_tiny(device):
    _test_torchvision_model("convnext_tiny", device=device)


@pytest.mark.torchvision
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_densenet121(device):
    _test_torchvision_model("densenet121", device=device)


@pytest.mark.torchvision
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_densenet161(device):
    _test_torchvision_model("densenet161", device=device)


@pytest.mark.torchvision
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_densenet169(device):
    _test_torchvision_model("densenet169", device=device)


@pytest.mark.torchvision
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_densenet201(device):
    _test_torchvision_model("densenet201", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_efficientnet_b0(device):
    _test_torchvision_model("efficientnet_b0", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_efficientnet_b1(device):
    _test_torchvision_model("efficientnet_b1", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_efficientnet_b2(device):
    _test_torchvision_model("efficientnet_b2", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_efficientnet_b3(device):
    _test_torchvision_model("efficientnet_b3", device=device)


@pytest.mark.torchvision
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_efficientnet_b4(device):
    _test_torchvision_model("efficientnet_b4", device=device)


@pytest.mark.torchvision
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_efficientnet_b5(device):
    _test_torchvision_model("efficientnet_b5", device=device)


@pytest.mark.torchvision
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_efficientnet_b6(device):
    _test_torchvision_model("efficientnet_b6", device=device)


@pytest.mark.torchvision
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_efficientnet_b7(device):
    _test_torchvision_model("efficientnet_b7", device=device)


@pytest.mark.torchvision
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_efficientnet_v2_l(device):
    _test_torchvision_model("efficientnet_v2_l", device=device)


@pytest.mark.torchvision
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_efficientnet_v2_m(device):
    _test_torchvision_model("efficientnet_v2_m", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_efficientnet_v2_s(device):
    _test_torchvision_model("efficientnet_v2_s", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_googlenet(device):
    _test_torchvision_model("googlenet", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_inception_v3(device):
    _test_torchvision_model("inception_v3", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_mnasnet0_5(device):
    _test_torchvision_model("mnasnet0_5", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_mnasnet0_75(device):
    _test_torchvision_model("mnasnet0_75", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_mnasnet1_0(device):
    _test_torchvision_model("mnasnet1_0", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_mnasnet1_3(device):
    _test_torchvision_model("mnasnet1_3", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_maxvit_t(device):
    _test_torchvision_model("maxvit_t", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_mobilenet_v2(device):
    _test_torchvision_model("mobilenet_v2", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_mobilenet_v3_large(device):
    _test_torchvision_model("mobilenet_v3_large", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_mobilenet_v3_small(device):
    _test_torchvision_model("mobilenet_v3_small", device=device)


@pytest.mark.torchvision
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_regnet_x_16gf(device):
    _test_torchvision_model("regnet_x_16gf", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_regnet_x_1_6gf(device):
    _test_torchvision_model("regnet_x_1_6gf", device=device)


@pytest.mark.torchvision
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_regnet_x_32gf(device):
    _test_torchvision_model("regnet_x_32gf", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_regnet_x_3_2gf(device):
    _test_torchvision_model("regnet_x_3_2gf", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_regnet_x_400mf(device):
    _test_torchvision_model("regnet_x_400mf", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_regnet_x_800mf(device):
    _test_torchvision_model("regnet_x_800mf", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_regnet_x_8gf(device):
    _test_torchvision_model("regnet_x_8gf", device=device)


@pytest.mark.torchvision
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_regnet_y_128gf(device):
    _test_torchvision_model("regnet_y_128gf", device=device)


@pytest.mark.torchvision
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_regnet_y_16gf(device):
    _test_torchvision_model("regnet_y_16gf", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_regnet_y_1_6gf(device):
    _test_torchvision_model("regnet_y_1_6gf", device=device)


@pytest.mark.torchvision
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_regnet_y_32gf(device):
    _test_torchvision_model("regnet_y_32gf", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_regnet_y_3_2gf(device):
    _test_torchvision_model("regnet_y_3_2gf", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_regnet_y_400mf(device):
    _test_torchvision_model("regnet_y_400mf", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_regnet_y_800mf(device):
    _test_torchvision_model("regnet_y_800mf", device=device)


@pytest.mark.torchvision
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_regnet_y_8gf(device):
    _test_torchvision_model("regnet_y_8gf", device=device)


@pytest.mark.torchvision
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_resnext101_32x8d(device):
    _test_torchvision_model("resnext101_32x8d", device=device)


@pytest.mark.torchvision
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_resnext101_64x4d(device):
    _test_torchvision_model("resnext101_64x4d", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_resnext50_32x4d(device):
    _test_torchvision_model("resnext50_32x4d", device=device)


@pytest.mark.torchvision
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_resnet101(device):
    _test_torchvision_model("resnet101", device=device)


@pytest.mark.torchvision
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_resnet152(device):
    _test_torchvision_model("resnet152", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_resnet18(device):
    _test_torchvision_model("resnet18", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_resnet34(device):
    _test_torchvision_model("resnet34", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_resnet50(device):
    _test_torchvision_model("resnet50", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_shufflenet_v2_x0_5(device):
    _test_torchvision_model("shufflenet_v2_x0_5", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_shufflenet_v2_x1_0(device):
    _test_torchvision_model("shufflenet_v2_x1_0", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_shufflenet_v2_x1_5(device):
    _test_torchvision_model("shufflenet_v2_x1_5", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_shufflenet_v2_x2_0(device):
    _test_torchvision_model("shufflenet_v2_x2_0", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_squeezenet1_0(device):
    _test_torchvision_model("squeezenet1_0", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_squeezenet1_1(device):
    _test_torchvision_model("squeezenet1_1", device=device)


@pytest.mark.torchvision
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_swin_b(device):
    _test_torchvision_model("swin_b", device=device)


@pytest.mark.torchvision
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_swin_s(device):
    _test_torchvision_model("swin_s", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_swin_t(device):
    _test_torchvision_model("swin_t", device=device)


@pytest.mark.torchvision
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_swin_v2_b(device):
    _test_torchvision_model("swin_v2_b", device=device)


@pytest.mark.torchvision
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_swin_v2_s(device):
    _test_torchvision_model("swin_v2_s", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_swin_v2_t(device):
    _test_torchvision_model("swin_v2_t", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_vgg11_bn(device):
    _test_torchvision_model("vgg11_bn", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_vgg11(device):
    _test_torchvision_model("vgg11", device=device)


@pytest.mark.torchvision
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_vgg13_bn(device):
    _test_torchvision_model("vgg13_bn", device=device)


@pytest.mark.torchvision
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_vgg13(device):
    _test_torchvision_model("vgg13", device=device)


@pytest.mark.torchvision
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_vgg16_bn(device):
    _test_torchvision_model("vgg16_bn", device=device)


@pytest.mark.torchvision
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_vgg16(device):
    _test_torchvision_model("vgg16", device=device)


@pytest.mark.torchvision
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_vgg19_bn(device):
    _test_torchvision_model("vgg19_bn", device=device)


@pytest.mark.torchvision
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_vgg19(device):
    _test_torchvision_model("vgg19", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_vit_b_16(device):
    _test_torchvision_model("vit_b_16", device=device)


@pytest.mark.torchvision
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_vit_b_32(device):
    _test_torchvision_model("vit_b_32", device=device)


@pytest.mark.torchvision
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_vit_h_14(device):
    _test_torchvision_model("vit_h_14", device=device)


@pytest.mark.torchvision
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_vit_l_16(device):
    _test_torchvision_model("vit_l_16", device=device)


@pytest.mark.torchvision
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_vit_l_32(device):
    _test_torchvision_model("vit_l_32", device=device)


@pytest.mark.torchvision
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_wide_resnet101_2(device):
    _test_torchvision_model("wide_resnet101_2", device=device)


@pytest.mark.torchvision
@pytest.mark.fast
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_wide_resnet50_2(device):
    _test_torchvision_model("wide_resnet50_2", device=device)


if __name__ == "__main__":
    DEBUG = True
    interp = _test_torchvision_model("convnext_base", device="cpu")
    print("flops: ", interp.flops)
    # interp.table.to_csv("torchvision_cuda.csv", index=False, sep="\t")
