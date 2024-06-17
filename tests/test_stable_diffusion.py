import pytest

import torch
from diffusers import DiffusionPipeline
from accelerate.utils import extract_model_from_parallel

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from model_inspector import ProfilingInterpreter


def test_stable_diffusion_xl_base_1_0_fp16():
    dtype = torch.float16
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16",
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pipe.to(device)

    latent = torch.randn(2, 4, 128, 128, device=device, dtype=dtype)
    t = torch.tensor(10., device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(2, 77, 2048, device=device, dtype=dtype)
    added_cond_kwargs = {
        "text_embeds": torch.randn(2, 1280, device=device, dtype=dtype),
        "time_ids": torch.tensor([[1024., 1024.,    0.,    0., 1024., 1024.], [1024., 1024.,    0.,    0., 1024., 1024.]], device=device, dtype=dtype),
    }

    interp = ProfilingInterpreter(pipe.unet, input_example=((latent, t, encoder_hidden_states, ), {"added_cond_kwargs": added_cond_kwargs}))
    _ = interp.run(latent, t, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs)
    table = interp.table
    return table


if __name__ == "__main__":
    table = test_stable_diffusion_xl_base_1_0_fp16()
    table.to_csv("sd.csv", index=False, sep="\t")
    table[table.valid].to_csv("sd_valid.csv", index=False, sep="\t")
    table[table.flops > 0].to_csv("sd_flops.csv", index=False, sep="\t")
    # import pdb; pdb.set_trace()
    print("done")
