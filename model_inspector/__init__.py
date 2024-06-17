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

from .inspect_functions import _calc_flops_conv, _calc_flops_attention


logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
except ImportError:
    logger.warning("tqdm not found. Install tqdm to see progress bars.")
    tqdm = lambda x, *args, **kwargs: x


# https://github.com/pytorch/pytorch/blob/b0282071c48860fcf8f4c1025bc207138173617b/torch/csrc/profiler/util.cpp#L405
FLOPABLE_OPS = {
    # manual
    "aten::_scaled_dot_product_flash_attention",
    "aten::convolution",

    # auto
    "aten::conv2d",
    "aten::mm",
    "aten::addmm",
    "aten::mul",
    "aten::add",
    "aten::bmm",
    "aten::baddbmm",
}

# aten::add.Tensor
# aten::mul.Tensor
# aten::div.Tensor

import shy; shy.err_hook()

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
        self.unsupported_function_types = []
        self.unsupported_method_types = []
        self.unsupported_module_types = []
        self._warning = False
        self._profiling = False
        self.profile_results = ProfileResults()
        self._torch_profiler: torch.profiler.profile | None = None
        self._timers = []
        self._flops_manual = defaultdict(int)

    def add_profile_result(self, profile_result: ProfileResult):
        self.profile_results.add(profile_result)

    def run(self, *args, **kwargs):
        self._profiling = True
        self._timers = []
        self._flops_manual = defaultdict(int)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
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
            if op_name == "aten::convolution":
                args, kwargs = self.fetch_args_kwargs_from_env(n)
                self._flops_manual[layer_name] += _calc_flops_conv(*args, **kwargs)
            if op_name == "aten::_scaled_dot_product_flash_attention":
                args, kwargs = self.fetch_args_kwargs_from_env(n)
                self._flops_manual[layer_name] += _calc_flops_attention(*args, **kwargs)
            with torch.profiler.record_function(layer_name), CudaTimer(layer_name) as timer:
                self._timers.append(timer)
                return super().run_node(n)
        else:
            return super().run_node(n)

    def get_cuda_time(self, name):
        time = 0
        for timer in self._timers:
            if timer.name == name:
                time += timer.elapsed_time()
        return time

    def get_flops(self, name):
        flops = self._flops_manual[name]
        return flops

    def get_cuda_time_accum(self, name):
        start_timer = self._timers[0].start
        end_timer = None
        for timer in self._timers:
            if timer.name == name:
                end_timer = timer.end
        if end_timer is None:
            return None
        return start_timer.elapsed_time(end_timer)

    @property
    def table(self):
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

                cuda_time = self.get_cuda_time(event.trace_name)
                new_cuda_time_accum = self.get_cuda_time_accum(event.trace_name)
                if new_cuda_time_accum is not None:
                    cuda_time_accum = new_cuda_time_accum

                flops = event.flops + self.get_flops(event.trace_name)

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
            df = self.calc_exact_flops(df)
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

    def calc_exact_flops(self, df):
        child_dict = defaultdict(list)
        parent_dict = defaultdict(list)
        index_to_flops = defaultdict(int)
        ids = df.id.to_numpy()
        shallow = df[["flops", "cpu_children", "sequence_nr"]].to_numpy()
        for index, (flops, cpu_children, sequence_nr) in enumerate(
            tqdm(shallow, total=len(df))
        ):
            index_to_flops[index] = flops
            children_index = np.isin(ids, cpu_children).nonzero()[0]
            children = shallow[children_index]
            for child_index, (child_flops, child_cpu_children, child_sequence_nr) in zip(
                children_index, children
            ):
                if child_sequence_nr == -1 or sequence_nr == -1:
                    parent_dict[child_index].append(index)
                    child_dict[index].append(child_index)

        queue = list()
        for index in df.index:
            if len(child_dict[index]) == 0 and len(parent_dict[index]) > 0:
                queue.append(index)

        print(f"# of leaf nodes: {len(queue)}")

        exact_flops_dict = copy.deepcopy(index_to_flops)
        while len(queue) > 0:
            node = queue.pop(0)
            node_flops = exact_flops_dict[node]

            if df.loc[node, "op"] in FLOPABLE_OPS:
                pass
            else:
                target_parent = None
                for parent in parent_dict[node]:
                    if df.loc[parent, "op"] in FLOPABLE_OPS:
                        target_parent = parent
                        break
                else:
                    target_parent = parent_dict[node][0]

                exact_flops_dict[target_parent] += node_flops
                exact_flops_dict[node] = 0

            for parent in parent_dict[node]:
                child_dict[parent].remove(node)
                if len(child_dict[parent]) == 0 and len(parent_dict[parent]) > 0:
                    queue.append(parent)

        total_flops = [exact_flops_dict.get(index, 0) for index in df.index]
        df["exact_flops"] = total_flops
        return df