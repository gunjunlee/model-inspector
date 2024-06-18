import pytest

import torch
from timm import create_model

from model_inspector import ProfilingInterpreter

from .common import REF_FLOPS
from .common import setup_logging


logger = setup_logging(__name__)


def _test_timm_model(model_name: str):
    model = create_model(model_name, pretrained=False)
    model.eval()
    args = (torch.randn(1, 3, 224, 224), )
    kwargs = {}
    input_example = (args, kwargs)
    interp = ProfilingInterpreter(model, input_example=input_example)
    _ = interp.run(*args, **kwargs)
    table = interp.table
    assert pytest.approx(table.exact_flops.sum(), rel=1e-1) == REF_FLOPS[model_name]
    return table


@pytest.mark.timm
@pytest.mark.fast
def test_resnet50():
    _test_timm_model("resnet50")


@pytest.mark.timm
@pytest.mark.fast
def test_vit_base_patch16_224():
    _test_timm_model("vit_base_patch16_224")


if __name__ == "__main__":
    table = _test_timm_model("resnet50")

