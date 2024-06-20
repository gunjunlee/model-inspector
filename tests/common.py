import pytest

import logging


# 2 flops = 1 mac
# https://pytorch.org/vision/main/models.html#table-of-all-available-classification-weights
_REF_FLOPS_VISION = {
    "alexnet": (0.71 * 1e+9 * 2, 224),
    "convnext_base": (15.36 * 1e+9 * 2, 224),
    "convnext_large": (34.36 * 1e+9 * 2, 224),
    "convnext_small": (8.68 * 1e+9 * 2, 224),
    "convnext_tiny": (4.46 * 1e+9 * 2, 224),
    "densenet121": (2.83 * 1e+9 * 2, 224),
    "densenet161": (7.73 * 1e+9 * 2, 224),
    "densenet169": (3.36 * 1e+9 * 2, 224),
    "densenet201": (4.29 * 1e+9 * 2, 224),
    "efficientnet_b0": (0.39 * 1e+9 * 2, 224),
    "efficientnet_b1": (0.69 * 1e+9 * 2, 240),
    "efficientnet_b2": (1.09 * 1e+9 * 2, 288),
    "efficientnet_b3": (1.83 * 1e+9 * 2, 300),
    "efficientnet_b4": (4.39 * 1e+9 * 2, 380),
    "efficientnet_b5": (10.27 * 1e+9 * 2, 456),
    "efficientnet_b6": (19.07 * 1e+9 * 2, 528),
    "efficientnet_b7": (37.75 * 1e+9 * 2, 600),
    "efficientnet_v2_l": (56.08 * 1e+9 * 2, 480),
    "efficientnet_v2_m": (24.58 * 1e+9 * 2, 480),
    "efficientnet_v2_s": (8.37 * 1e+9 * 2, 384),
    "googlenet": (1.50 * 1e+9 * 2, 224),
    "inception_v3": (5.71 * 1e+9 * 2, 299),
    "mnasnet0_5": (0.10 * 1e+9 * 2, 224),
    "mnasnet0_75": (0.21 * 1e+9 * 2, 224),
    "mnasnet1_0": (0.31 * 1e+9 * 2, 224),
    "mnasnet1_3": (0.53 * 1e+9 * 2, 224),
    "maxvit_t": (5.56 * 1e+9 * 2, 224),
    "mobilenet_v2": (0.30 * 1e+9 * 2, 224),
    "mobilenet_v3_large": (0.22 * 1e+9 * 2, 224),
    "mobilenet_v3_small": (0.06 * 1e+9 * 2, 224),
    "regnet_x_16gf": (15.94 * 1e+9 * 2, 224),
    "regnet_x_1_6gf": (1.60 * 1e+9 * 2, 224),
    "regnet_x_32gf": (31.74 * 1e+9 * 2, 224),
    "regnet_x_3_2gf": (3.18 * 1e+9 * 2, 224),
    "regnet_x_400mf": (0.41 * 1e+9 * 2, 224),
    "regnet_x_800mf": (0.80 * 1e+9 * 2, 224),
    "regnet_x_8gf": (8.00 * 1e+9 * 2, 224),
    "regnet_y_128gf": (374.57 * 1e+9 * 2, 384),
    "regnet_y_16gf": (15.91 * 1e+9 * 2, 224),
    "regnet_y_1_6gf": (1.61 * 1e+9 * 2, 224),
    "regnet_y_32gf": (32.28 * 1e+9 * 2, 224),
    "regnet_y_3_2gf": (3.18 * 1e+9 * 2, 224),
    "regnet_y_400mf": (0.40 * 1e+9 * 2, 224),
    "regnet_y_800mf": (0.83 * 1e+9 * 2, 224),
    "regnet_y_8gf": (8.47 * 1e+9 * 2, 224),
    "resnext101_32x8d": (16.41 * 1e+9 * 2, 224),
    "resnext101_64x4d": (15.46 * 1e+9 * 2, 224),
    "resnext50_32x4d": (4.23 * 1e+9 * 2, 224),
    "resnet101": (7.80 * 1e+9 * 2, 224),
    "resnet152": (11.51 * 1e+9 * 2, 224),
    "resnet18": (1.81 * 1e+9 * 2, 224),
    "resnet34": (3.66 * 1e+9 * 2, 224),
    "resnet50": (4.09 * 1e+9 * 2, 224),
    "shufflenet_v2_x0_5": (0.04 * 1e+9 * 2, 224),
    "shufflenet_v2_x1_0": (0.14 * 1e+9 * 2, 224),
    "shufflenet_v2_x1_5": (0.30 * 1e+9 * 2, 224),
    "shufflenet_v2_x2_0": (0.58 * 1e+9 * 2, 224),
    "squeezenet1_0": (0.82 * 1e+9 * 2, 224),
    "squeezenet1_1": (0.35 * 1e+9 * 2, 224),
    "swin_b": (15.43 * 1e+9 * 2, 224),
    "swin_s": (8.74 * 1e+9 * 2, 224),
    "swin_t": (4.49 * 1e+9 * 2, 224),
    "swin_v2_b": (20.32 * 1e+9 * 2, 256),
    "swin_v2_s": (11.55 * 1e+9 * 2, 256),
    "swin_v2_t": (5.94 * 1e+9 * 2, 256),
    "vgg11_bn": (7.61 * 1e+9 * 2, 224),
    "vgg11": (7.61 * 1e+9 * 2, 224),
    "vgg13_bn": (11.31 * 1e+9 * 2, 224),
    "vgg13": (11.31 * 1e+9 * 2, 224),
    "vgg16_bn": (15.47 * 1e+9 * 2, 224),
    "vgg16": (15.47 * 1e+9 * 2, 224),
    "vgg19_bn": (19.63 * 1e+9 * 2, 224),
    "vgg19": (19.63 * 1e+9 * 2, 224),
    "vit_b_16": (17.56 * 1e+9 * 2, 224),
    "vit_b_32": (4.41 * 1e+9 * 2, 224),
    "vit_h_14": (1016.72 * 1e+9 * 2, 518),
    "vit_l_16": (61.55 * 1e+9 * 2, 224),
    "vit_l_32": (15.38 * 1e+9 * 2, 224),
    "wide_resnet101_2": (22.75 * 1e+9 * 2, 224),
    "wide_resnet50_2": (11.40 * 1e+9 * 2, 224),
}

_REF_FLOPS_VISION["vit_base_patch16_224"] = _REF_FLOPS_VISION["vit_b_16"]


def get_vision_spec(name: str):
    spec = _REF_FLOPS_VISION[name]
    if isinstance(spec, tuple):
        return spec
    return spec, 224


# just for making _REF_FLOPS_VISION
def _get_torchvision_models():
    import inspect
    import torchvision
    for attr in dir(torchvision.models):
        if attr.endswith("_Weights"):
            model_name = attr[:-len("_Weights")].lower()

            weight_configs = getattr(torchvision.models, attr)
            weight_attr_name = None
            if hasattr(weight_configs, "IMAGENET1K_V1"):
                weight_attr_name = "IMAGENET1K_V1"
            else:
                for weight_attr in dir(weight_configs):
                    if weight_attr.startswith("IMAGENET"):
                        weight_attr_name = weight_attr
                        break
                else:
                    print(f"skip {model_name}")
                    continue
            weight_config = getattr(weight_configs, weight_attr_name)
            flops = weight_config.meta["_ops"]
            signature = inspect.signature(weight_config.transforms).bind_partial()
            signature.apply_defaults()
            image_size = signature.arguments["crop_size"]
            print(f"\"{model_name}\": ({flops:.2f} * 1e+9 * 2, {image_size}),")

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

    formatter = logging.Formatter('%(asctime)s:%(module)s:%(levelname)s:%(message)s', '%Y-%m-%d %H:%M:%S')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)

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


def get_available_devices():
    import torch
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


if __name__ == "__main__":
    _get_torchvision_models()
