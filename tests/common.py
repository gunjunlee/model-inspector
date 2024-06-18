import pytest

import logging


# 2 flops = 1 mac
# https://pytorch.org/vision/main/models.html#table-of-all-available-classification-weights
REF_FLOPS = {
    "alexnet": 0.71 * 1e+9 * 2,
    "convnext_base": 15.36 * 1e+9 * 2,
    "convnext_large": 34.36 * 1e+9 * 2,
    "convnext_small": 8.68 * 1e+9 * 2,
    "convnext_tiny": 4.46 * 1e+9 * 2,
    "densenet121": 2.83 * 1e+9 * 2,
    "densenet161": 7.73 * 1e+9 * 2,
    "densenet169": 3.36 * 1e+9 * 2,
    "densenet201": 4.29 * 1e+9 * 2,
    "efficientnet_b0": 0.39 * 1e+9 * 2,
    "efficientnet_b1": 0.69 * 1e+9 * 2,
    "efficientnet_b2": 1.09 * 1e+9 * 2,
    "efficientnet_b3": 1.83 * 1e+9 * 2,
    "efficientnet_b4": 4.39 * 1e+9 * 2,
    "efficientnet_b5": 10.27 * 1e+9 * 2,
    "efficientnet_b6": 19.07 * 1e+9 * 2,
    "efficientnet_b7": 37.75 * 1e+9 * 2,
    "efficientnet_v2_l": 56.08 * 1e+9 * 2,
    "efficientnet_v2_m": 24.58 * 1e+9 * 2,
    "efficientnet_v2_s": 8.37 * 1e+9 * 2,
    "googlenet": 1.50 * 1e+9 * 2,
    "inception_v3": 5.71 * 1e+9 * 2,
    "mnasnet0_5": 0.10 * 1e+9 * 2,
    "mnasnet0_75": 0.21 * 1e+9 * 2,
    "mnasnet1_0": 0.31 * 1e+9 * 2,
    "mnasnet1_3": 0.53 * 1e+9 * 2,
    "maxvit_t": 5.56 * 1e+9 * 2,
    "mobilenet_v2": 0.30 * 1e+9 * 2,
    "mobilenet_v3_large": 0.22 * 1e+9 * 2,
    "mobilenet_v3_small": 0.06 * 1e+9 * 2,
    "regnet_x_16gf": 15.94 * 1e+9 * 2,
    "regnet_x_1_6gf": 1.60 * 1e+9 * 2,
    "regnet_x_32gf": 31.74 * 1e+9 * 2,
    "regnet_x_3_2gf": 3.18 * 1e+9 * 2,
    "regnet_x_400mf": 0.41 * 1e+9 * 2,
    "regnet_x_800mf": 0.80 * 1e+9 * 2,
    "regnet_x_8gf": 8.00 * 1e+9 * 2,
    "regnet_y_128gf": 127.52 * 1e+9 * 2,
    "regnet_y_16gf": 15.91 * 1e+9 * 2,
    "regnet_y_1_6gf": 1.61 * 1e+9 * 2,
    "regnet_y_32gf": 32.28 * 1e+9 * 2,
    "regnet_y_3_2gf": 3.18 * 1e+9 * 2,
    "regnet_y_400mf": 0.40 * 1e+9 * 2,
    "regnet_y_800mf": 0.83 * 1e+9 * 2,
    "regnet_y_8gf": 8.47 * 1e+9 * 2,
    "resnext101_32x8d": 16.41 * 1e+9 * 2,
    "resnext101_64x4d": 15.46 * 1e+9 * 2,
    "resnext50_32x4d": 4.23 * 1e+9 * 2,
    "resnet101": 7.80 * 1e+9 * 2,
    "resnet152": 11.51 * 1e+9 * 2,
    "resnet18": 1.81 * 1e+9 * 2,
    "resnet34": 3.66 * 1e+9 * 2,
    "resnet50": 4.09 * 1e+9 * 2,
    "shufflenet_v2_x0_5": 0.04 * 1e+9 * 2,
    "shufflenet_v2_x1_0": 0.14 * 1e+9 * 2,
    "shufflenet_v2_x1_5": 0.30 * 1e+9 * 2,
    "shufflenet_v2_x2_0": 0.58 * 1e+9 * 2,
    "squeezenet1_0": 0.82 * 1e+9 * 2,
    "squeezenet1_1": 0.35 * 1e+9 * 2,
    "swin_b": 15.43 * 1e+9 * 2,
    "swin_s": 8.74 * 1e+9 * 2,
    "swin_t": 4.49 * 1e+9 * 2,
    "swin_v2_b": 20.32 * 1e+9 * 2,
    "swin_v2_s": 11.55 * 1e+9 * 2,
    "swin_v2_t": 5.94 * 1e+9 * 2,
    "vgg11_bn": 7.61 * 1e+9 * 2,
    "vgg11": 7.61 * 1e+9 * 2,
    "vgg13_bn": 11.31 * 1e+9 * 2,
    "vgg13": 11.31 * 1e+9 * 2,
    "vgg16_bn": 15.47 * 1e+9 * 2,
    "vgg16": 15.47 * 1e+9 * 2,
    "vgg19_bn": 19.63 * 1e+9 * 2,
    "vgg19": 19.63 * 1e+9 * 2,
    "vit_b_16": 17.56 * 1e+9 * 2,
    "vit_base_patch16_224": 17.56 * 1e+9 * 2,
    "vit_b_32": 4.41 * 1e+9 * 2,
    "vit_h_14": 167.29 * 1e+9 * 2,
    "vit_l_16": 61.55 * 1e+9 * 2,
    "vit_l_32": 15.38 * 1e+9 * 2,
    "wide_resnet101_2": 22.75 * 1e+9 * 2,
    "wide_resnet50_2": 11.00 * 1e+9 * 2,
}

# for k, v in REF_FLOPS.items():
#     print(
# f"""
# @pytest.mark.torchvision
# @pytest.mark.fast
# def test_{k}():
#     _test_torchvision_model("{k}")
# """
#     )


def setup_logging(name):
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("torch").setLevel(logging.WARNING)
    # ignore deprecated warnings
    logging.getLogger("torch").addFilter(lambda record: "is deprecated" not in record.getMessage())

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.addHandler(console_handler)

    return logger


def print_flops(flops):
    if flops > 1e+12:
        return f"{flops / 1e+12:.2f}T"
    if flops > 1e+9:
        return f"{flops / 1e+9:.2f}G"
    if flops > 1e+6:
        return f"{flops / 1e+6:.2f}M"
    if flops > 1e+3:
        return f"{flops / 1e+3:.2f}K"
    return f"{flops:d}"


def check_flops(model_name, calc_flops, ref_flops):
    assert pytest.approx(calc_flops, rel=1e-1) == ref_flops, \
        f"[{model_name}] " \
        f"calc flops: {print_flops(calc_flops)}, " \
        f"ref flops: {print_flops(ref_flops)} (x{ref_flops / calc_flops:.1f})"
