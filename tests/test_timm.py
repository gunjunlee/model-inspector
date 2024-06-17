import pytest

import torch
from timm import create_model

from model_inspector import ProfilingInterpreter


# ref FLOPS from https://github.com/sovrasov/flops-counter.pytorch/tree/master
REF_FLOPS = {
    "resnet50": 4.13e+9 * 2  # 2 flops = 1 mac
}


def _test_resnet50():
    model = create_model("resnet50", pretrained=False)
    model.eval()
    args = (torch.randn(1, 3, 224, 224), )
    kwargs = {}
    input_example = (args, kwargs)
    interp = ProfilingInterpreter(model, input_example=input_example)
    _ = interp.run(*args, **kwargs)
    table = interp.table
    assert pytest.approx(table.exact_flops.sum(), rel=1e-1) == REF_FLOPS["resnet50"]
    return table


# ref FLOPS from https://github.com/sovrasov/flops-counter.pytorch/tree/master
def _test_vit_base_patch16_224():
    model = create_model("vit_base_patch16_224", pretrained=False)
    model.eval()
    args = (torch.randn(1, 3, 224, 224), )
    kwargs = {}
    input_example = (args, kwargs)
    interp = ProfilingInterpreter(model, input_example=input_example)
    _ = interp.run(*args, **kwargs)
    table = interp.table
    assert pytest.approx(table.exact_flops.sum(), rel=1e-1) == REF_FLOPS["resnet50"]
    return table


@pytest.mark.timm
def test_resnet50():
    _test_resnet50()


@pytest.mark.timm
def test_vit_base_patch16_224():
    _test_vit_base_patch16_224()


if __name__ == "__main__":
    table = _test_resnet50()
    _test_vit_base_patch16_224()
