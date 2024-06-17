__version__ = "0.0.1"

import torch
from torch.fx import Interpreter

import pandas as pd

import logging
import inspect
import operator
import functools
from collections import OrderedDict
from typing import List, Tuple, Dict, Any, Union

from .inspect_functions import _INSPECTABLE_FUNCTIONS


logger = logging.getLogger(__name__)


import shy; shy.err_hook()

def _inspect_none(profiler, node):
    return None


_INSPECTABLE_METHODS = {
    "flatten": _inspect_none,
    "view": _inspect_none,
    "transpose": _inspect_none,
    "expand": _inspect_none,
    "reshape": _inspect_none,
    "permute": _inspect_none,
    "contiguous": _inspect_none,
    "size": _inspect_none,
    "squeeze": _inspect_none,
    "unbind": _inspect_none,
}


_INSPECTABLE_MODULES = {
    # torch.nn.Conv2d: _inspect_module_conv2d,
    # torch.nn.Linear: _inspect_module_linear,
    # torch.nn.Identity: _inspect_none,
    # torch.nn.Dropout: _inspect_none,
    # torch.nn.LayerNorm: _inspect_none,
    # torch.nn.ReLU: _inspect_none,
    # torch.nn.GELU: _inspect_none,
}


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


class CustomTracer(torch.fx.Tracer):
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str):
        ret = super().is_leaf_module(m, module_qualified_name)
        print(m.__class__.__name__, module_qualified_name, ret)
        return ret


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


# class DecomposeTransformer(torch.fx.Transformer):
#     def call_function(self, target, args, kwargs):
#         if isinstance(target, torch._ops.OpOverload):
#             assert len(target.py_kernels) == 1, "Only one kernel is supported"
#             op = list(target.py_kernels.values())[0]
#             return op(*args, **kwargs)
#         return super().call_function(target, args, kwargs)


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
#             module = torch.jit.trace(module, example_kwarg_inputs=fixed_forward_arguments, strict=False)
#             module_arguments = [arg.name for arg in module.forward.schema.arguments]

#             cls_def_str = f"""
# class _Wrapper_impl(torch.nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model

#     def forward({", ".join(module_arguments)}):
#         return self.model({", ".join(module_arguments[1:])})
#             """
#             exec(cls_def_str, globals())
#             _Wrapper = eval("_Wrapper_impl")
#             module = _Wrapper(module)
#         else:
#             module_arguments = list(inspect.signature(module.forward).parameters.keys())

#         self.model = module
#         self.module_arguments = module_arguments
#         traced_graph = CustomTracer().trace(self.model)
#         gm = torch.fx.GraphModule(self.model, traced_graph)
        self.exported_model = torch.export.export(module, (), kwargs=fixed_forward_arguments)
#         model_arguments = [p for p in self.exported_model.graph_signature.input_specs if p.arg.name in self.exported_model.graph_signature.user_inputs]
#         self.model_arguments = model_arguments
        # gm = DecomposeTransformer(self.exported_model.module()).transform()
        # super().__init__(gm)
        super().__init__(self.exported_model.module())
        self.model_arguments = list(inspect.signature(self.module.forward).parameters)
        # self.exported_model(**fixed_forward_arguments)
        self.unsupported_function_types = []
        self.unsupported_method_types = []
        self.unsupported_module_types = []
        self._warning = False
        self._profiling = False
        self.profile_results = ProfileResults()

    def add_profile_result(self, profile_result: ProfileResult):
        self.profile_results.add(profile_result)

    def run(self, *args, **kwargs):
        self._profiling = True

        forward_parameters = args
        for name in self.model_arguments[len(args):]:
            if name not in kwargs:
                raise ValueError(f"Missing argument {name}")
            forward_parameters += (kwargs[name],)

        ret = super().run(*forward_parameters)
        if self._warning:
            logger.warning("The model contains unsupported operations.")
            for f in self.unsupported_function_types:
                logger.warning(f)
            for f in self.unsupported_method_types:
                logger.warning(f)
            for f in list(set(self.unsupported_module_types)):
                logger.warning(f)
            import pdb; pdb.set_trace()
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
                        assert len(op.py_kernels) == 1, "Only one kernel is supported"
                        op = list(op.py_kernels.values())[0]
                    if op in _INSPECTABLE_FUNCTIONS:
                        self.add_profile_result(_INSPECTABLE_FUNCTIONS[op](self, n))
                    elif n.target not in self.unsupported_function_types:
                        print(n, n.target, n.op, self.fetch_args_kwargs_from_env(n), op.__module__, op)
                        import pdb; pdb.set_trace()
                        self.unsupported_function_types.append(n.target)
                        self._warning = True
                elif n.op == "call_method":
                    if n.target in _INSPECTABLE_METHODS:
                        self.add_profile_result(_INSPECTABLE_METHODS[n.target](self, n))
                    elif n.target not in self.unsupported_method_types:
                        self.unsupported_method_types.append(n.target)
                        self._warning = True
                elif n.op == "call_module":
                    module = self.fetch_attr(n.target).__class__
                    if module in _INSPECTABLE_MODULES:
                        self.add_profile_result(_INSPECTABLE_MODULES[module](self, n))
                    elif module not in self.unsupported_module_types:
                        self.unsupported_module_types.append(module)
                        self._warning = True
                elif n.op == "placeholder":
                    pass
                elif n.op == "get_attr":
                    pass
                elif n.op == "output":
                    pass
                else:
                    raise NotImplementedError(f"Unsupported op {n.op}")
        return super().run_node(n)

    def print_stats(self):
        for k, v in self.profile_results.profile_results.items():
            print(k, v)

    @property
    def table(self):
        return self.profile_results.table
