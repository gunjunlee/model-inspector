# https://github.com/pytorch/pytorch/blob/b0282071c48860fcf8f4c1025bc207138173617b/torch/csrc/profiler/util.cpp#L542
__version__ = "0.0.1"

import torch
from torch.fx import Interpreter

import numpy as np
import pandas as pd

import copy
import logging
import inspect
from collections import defaultdict
from collections import OrderedDict
from typing import List, Tuple, Dict, Any, Union

from .inspect_functions import FLOPABLE_OPS
from .inspect_functions import get_flops_counter


logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
except ImportError:
    logger.warning("tqdm not found. Install tqdm to see progress bars.")
    tqdm = lambda x, *args, **kwargs: x


_INSPECT_PREFIX = "_MOD_INS:"

class ProfileResult:
    def __init__(self, node, op_name, op_type, hits=1, mads=0, ops=0, param_size=0, description=""):
        self.node = node
        self.op_name = op_name
        self.op_type = op_type
        self.hits = hits
        self.mads = mads
        self.ops = ops
        self.param_size = param_size
        self.description = ""

    def __str__(self):
        return f"[{str(hash(self.node))[:6]}] {self.op_name}: {self.op_type} hits={self.hits}, mads={self.mads}, ops={self.ops}, param_size={self.param_size}, description={self.description}"

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        self.hits += other.hits
        self.mads += other.mads
        self.ops += other.ops
        self.param_size += other.param_size
        if self.description != other.description:
            logger.warning(f"Description mismatch: {self.description} != {other.description}")
        return self

    def todict(self):
        return {
            "node": self.node,
            "op_name": self.op_name,
            "op_type": self.op_type,
            "hits": self.hits,
            "mads": self.mads,
            "ops": self.ops,
            "param_size": self.param_size,
            "description": self.description,
        }

class ProfileResults:
    def __init__(self):
        self.profile_results: Dict[torch.fx.Node, ProfileResult] = OrderedDict()

    def add(self, profile_result: ProfileResult | None):
        if profile_result is None:
            return
        if profile_result.node in self.profile_results:
            self.profile_results[profile_result.node] += profile_result
        else:
            self.profile_results[profile_result.node] = profile_result

    @property
    def table(self) -> pd.DataFrame:
        return pd.DataFrame([v.todict() for v in self.profile_results.values()])


class CudaTimer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        return self

    def __exit__(self, type, value, traceback):
        self.end.record()

    def elapsed_time(self):
        return self.start.elapsed_time(self.end)


class ProfilingInterpreter(Interpreter):
    def __init__(self, module: torch.nn.Module, parent=None, input_example=None):
        if parent is not None:
            self.parent = parent
        else:
            self.parent = self

        if input_example is not None:
            assert len(input_example) == 2, "Input example should be a tuple of args and kwargs"
            args, kwargs = input_example
            assert isinstance(args, tuple), "Args should be a tuple"
            assert isinstance(kwargs, dict), "Kwargs should be a dict"

            fixed_forward_arguments = inspect.signature(module.forward).bind(*args, **kwargs).arguments
            exported_model = torch.export.export(module, (), kwargs=fixed_forward_arguments)
            gm = exported_model.module()
        else:
            gm = torch.fx.symbolic_trace(module)
        super().__init__(gm)
        self.model_arguments = list(inspect.signature(self.module.forward).parameters)
        self.init()

    def add_profile_result(self, profile_result: ProfileResult):
        self.profile_results.add(profile_result)

    def init(self):
        self.unsupported_function_types = []
        self.unsupported_method_types = []
        self.unsupported_module_types = []
        self._warning = False
        self._profiling = False
        self.profile_results = ProfileResults()
        self._torch_profiler: torch.profiler.profile | None = None
        self._timers = []
        self._flops_manual = defaultdict(int)
        self._table = None
        self._is_cuda_op = defaultdict(lambda : False)

    def run(self, *args, **kwargs):
        self.init()
        self._profiling = True

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=False,
            with_modules=True,
        ) as p:
            self._torch_profiler = p

            forward_parameters = args
            for name in self.model_arguments[len(args):]:
                if name not in kwargs:
                    raise ValueError(f"Missing argument {name}")
                forward_parameters += (kwargs[name],)

            input_shapes = ", ".join(str(arg.shape) for arg in forward_parameters if isinstance(arg, torch.Tensor))
            layer_name = f"{_INSPECT_PREFIX}TOTAL/TOTAL/{input_shapes}"
            with torch.profiler.record_function(layer_name), CudaTimer(layer_name) as timer:
                self._timers.append(timer)
                ret = super().run(*forward_parameters)

        if self._warning:
            logger.warning("The model contains unsupported operations.")
            for f in self.unsupported_function_types:
                logger.warning(f)
            for f in self.unsupported_method_types:
                logger.warning(f)
            for f in list(set(self.unsupported_module_types)):
                logger.warning(f)
            self._warning = False
        self._profiling = False

        return ret

    def run_node(self, n: torch.fx.Node):
        if self._profiling:
            logger.debug(f"Running node {n.name}")
            with self._set_current_node(n):
                if n.op == "call_function":
                    op = n.target
                    if isinstance(op, torch._ops.OpOverload):
                        op_name = op.name()
                    else:
                        op_name = op.__name__
                elif n.op == "call_method":
                    op = n.target
                    op_name = op.__name__
                elif n.op == "call_module":
                    op = n.target
                    op_name = op.__name__
                elif n.op == "placeholder":
                    op_name = op = None
                elif n.op == "get_attr":
                    op_name = op = None
                elif n.op == "output":
                    op_name = op = None
                else:
                    raise NotImplementedError(f"Unsupported op {n.op}")

            if op_name is not None:
                node_name = list(n.meta.get("nn_module_stack", [""]))[-1]
                args, kwargs = self.fetch_args_kwargs_from_env(n)
                input_shapes = ", ".join(str(arg.shape) for arg in args if isinstance(arg, torch.Tensor))
                layer_name = f"{_INSPECT_PREFIX}{node_name}/{op_name}/{input_shapes}"
                with torch.profiler.record_function(layer_name):
                    if op_name in FLOPABLE_OPS:
                        args, kwargs = self.fetch_args_kwargs_from_env(n)
                        self._flops_manual[layer_name] += get_flops_counter(op_name)(*args, **kwargs)
                        if args[0].device.type == "cuda":
                            with CudaTimer(layer_name) as timer:
                                self._is_cuda_op[layer_name] = True
                                self._timers.append(timer)
                                return super().run_node(n)
                    return super().run_node(n)

        return super().run_node(n)

    def get_cuda_time(self, name, device):
        if device == "CUDA":
            time = 0
            for timer in self._timers:
                if timer.name == name:
                    time += timer.elapsed_time()
            return time
        elif device == "CPU":
            return 0
        else:
            raise ValueError(f"Unknown device {device}")

    def get_flops(self, name, device):
        if device == "CUDA":
            val = self._flops_manual[name]
            self._flops_manual[name] = 0  # group convolutions can be counted multiple times in the same layer (maybe torch profiler bug)
            return val
        elif device == "CPU":
            if self._is_cuda_op[name]:
                return 0
            return self._flops_manual[name]
        else:
            raise ValueError(f"Unknown device {device}")

    def get_cuda_time_accum(self, name, device):
        if device == "CUDA":
            start_timer = self._timers[0].start
            end_timer = None
            for timer in self._timers:
                if timer.name == name:
                    end_timer = timer.end
            if end_timer is None:
                return None
            return start_timer.elapsed_time(end_timer)
        elif device == "CPU":
            return None
        else:
            raise ValueError(f"Unknown device {device}")

    @property
    def flops(self):
        df = self.table
        return df.flops.sum()

    @property
    def cuda_flops(self):
        df = self.table
        return df[df.device == "CUDA"].flops.sum()

    @property
    def table(self):
        if self._table is not None:
            return self._table
        self._table = self._get_table()
        return self._table

    @property
    def flops_table(self):
        df = self.table
        return df[df.flops > 0]

    def _get_table(self):
        if self._torch_profiler is not None:
            torch.cuda.synchronize()
            datas = []
            cuda_time_accum = 0
            for event in self._torch_profiler.events():
                if event.trace_name is not None and event.trace_name.startswith(_INSPECT_PREFIX):
                    name, op, input_shapes = event.trace_name.split("/")
                    valid = True
                else:
                    name, op, input_shapes = event.name, str("none"), str("none")
                    valid = False

                device = event.device_type.name
                cuda_time = self.get_cuda_time(event.trace_name, device)
                new_cuda_time_accum = self.get_cuda_time_accum(event.trace_name, device)
                if new_cuda_time_accum is not None:
                    cuda_time_accum = new_cuda_time_accum

                flops = self.get_flops(event.trace_name, device)

                data = {
                    "name": name,
                    "op": op,
                    "input_shapes": input_shapes,
                    "id": event.id,
                    "cpu_time": event.cpu_time,
                    "cpu_time_total": event.cpu_time_total,
                    "cpu_memory_usage": event.cpu_memory_usage,
                    "cuda_time": event.cuda_time,
                    "cuda_time_total": event.cuda_time_total,
                    "cuda_time_manual": cuda_time,
                    "cuda_time_accum_manual": cuda_time_accum,
                    "cuda_memory_usage": event.cuda_memory_usage,
                    "flops": flops,
                    "start_time": event.time_range.start,
                    "end_time": event.time_range.end,
                    "cpu_children": [c.id for c in event.cpu_children],
                    "scope": event.scope,
                    "valid": valid,
                    "thread": event.thread,
                    "fwd_thread": event.fwd_thread,
                    "device": event.device_type.name,
                    "sequence_nr": event.sequence_nr,
                }
                datas.append(data)
            df = pd.DataFrame(datas)
            df["cpu_time_rate"] = df["cpu_time_total"] / df["cpu_time_total"].max()
            df["cuda_time_rate"] = df["cuda_time_manual"] / df["cuda_time_manual"].max()

            df.loc[df.op == "TOTAL", "cpu_children"] = None
            df["flops_accum"] = df["flops"].cumsum()
            df["name"] = df["name"].map(
                lambda x: x[len(_INSPECT_PREFIX):] if x.startswith(_INSPECT_PREFIX) else x
            )
            return df
        else:
            logger.warning("No profiler results available")
            return None
