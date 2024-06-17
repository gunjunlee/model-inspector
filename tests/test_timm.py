import pytest

import torch
from tests.test_timm import create_model

from model_inspector import ProfilingInterpreter


def test_vit_base_patch16_224():
    model = create_model("vit_base_patch16_224", pretrained=True)
    model.eval()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as p:
        y = model(torch.randn(1, 3, 224, 224))
    print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    
    # interp = ProfilingInterpreter(model)
    # y = interp.run(torch.randn(1, 3, 224, 224))
    # interp.print_stats()
    # interp.table

