# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Colocated Python function API implementation."""

from __future__ import annotations

import dataclasses
import inspect
import random
import threading
from typing import Any, Callable, Sequence

import jax
from jax._src import api
from jax._src import tree_util
from jax._src.lib import xla_client as xc
from jax._src.traceback_util import api_boundary
from jax._src.util import wraps
from jax.experimental.colocated_python import func_backend
from jax.experimental.colocated_python.serialization import deserialize_specs, make_specs_for_serialized_specs, serialize_specs

ShapeDtypeStructTree = Any  # PyTree[ShapeDtypeStruct]


@dataclasses.dataclass(frozen=True, slots=True)
class FunctionInfo:
  """User function wrapped by colocated_python."""

  fun: Callable[..., Any]
  fun_sourceinfo: str | None
  fun_signature: inspect.Signature | None


@dataclasses.dataclass(frozen=True, slots=True)
class Specialization:
  """Specialization for a colocated_python function."""

  in_specs_treedef: tree_util.PyTreeDef | None = None
  in_specs_leaves: tuple[api.ShapeDtypeStruct, ...] | None = None
  out_specs_fn: Callable[..., Any] | None = None
  out_specs_treedef: tree_util.PyTreeDef | None = None
  out_specs_leaves: tuple[api.ShapeDtypeStruct, ...] | None = None
  devices: xc.DeviceList | None = None


def _tree_flatten(x: Any) -> tuple[tree_util.PyTreeDef, tuple[Any, ...]]:
  """Same as tree_util.tree_flatten, but with different result formatting.

  It returns (treedef, leaves) instead of (leaves, treedef) for consistency with
  tree_unflatten(). leaves are a tuple for easier use as a cache key.
  """
  leaves, treedef = tree_util.tree_flatten(x)
  return treedef, tuple(leaves)


def _apply_specialization(
    base_specialization: Specialization,
    in_specs_treedef: tree_util.PyTreeDef | None = None,
    in_specs_leaves: tuple[api.ShapeDtypeStruct, ...] | None = None,
    out_specs_fn: Callable[..., Any] | None = None,
    out_specs_treedef: tree_util.PyTreeDef | None = None,
    out_specs_leaves: tuple[api.ShapeDtypeStruct, ...] | None = None,
    devices: Sequence[jax.Device] | xc.DeviceList | None = None,
) -> Any:
  """Applies extra specialization to the base specialization."""
  new_in_specs_treedef = base_specialization.in_specs_treedef
  new_in_specs_leaves = base_specialization.in_specs_leaves
  new_out_specs_fn = base_specialization.out_specs_fn
  new_out_specs_treedef = base_specialization.out_specs_treedef
  new_out_specs_leaves = base_specialization.out_specs_leaves
  new_devices = base_specialization.devices

  if in_specs_treedef is not None:
    if base_specialization.in_specs_treedef is not None:
      raise ValueError("in_specs already specified")
    new_in_specs_treedef = in_specs_treedef
  if in_specs_leaves is not None:
    if base_specialization.in_specs_leaves is not None:
      raise ValueError("in_specs already specified")
    new_in_specs_leaves = in_specs_leaves

  if out_specs_fn is not None:
    if base_specialization.out_specs_fn is not None:
      raise ValueError("out_specs_fn already specified")
    new_out_specs_fn = out_specs_fn

  if out_specs_treedef is not None:
    if base_specialization.out_specs_treedef is not None:
      raise ValueError("out_specs already specified")
    new_out_specs_treedef = out_specs_treedef
  if out_specs_leaves is not None:
    if base_specialization.out_specs_leaves is not None:
      raise ValueError("out_specs already specified")
    new_out_specs_leaves = out_specs_leaves

  if devices is not None:
    if base_specialization.devices is not None:
      raise ValueError("devices already specified")
    if isinstance(devices, xc.DeviceList):
      new_devices = devices
    else:
      new_devices = xc.DeviceList(tuple(devices))

  return Specialization(
      in_specs_treedef=new_in_specs_treedef,
      in_specs_leaves=new_in_specs_leaves,
      out_specs_fn=new_out_specs_fn,
      out_specs_treedef=new_out_specs_treedef,
      out_specs_leaves=new_out_specs_leaves,
      devices=new_devices,
  )


def _get_spec(x: Any) -> api.ShapeDtypeStruct:
  """Extracts a spec for a value, which must be a JAX Array."""
  # TODO(hyeontaek): Allow Python values and automatically apply `shard_arg`
  # with a suitable sharding and layout.
  if not isinstance(x, jax.Array):
    raise ValueError(
        "colocated_python only supports jax.Array as input and output, but got"
        f" {type(x)}."
    )
  return api.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype, sharding=x.sharding)


def _get_specs(x: tuple[Any, ...]) -> tuple[api.ShapeDtypeStruct, ...]:
  """Extracts a spec for values, which must be JAX Arrays."""
  # TODO(hyeontaek): Allow Python values and apply `shard_arg` with a suitable
  # sharding and layout.
  return tuple(_get_spec(x) for x in x)


def _infer_devices_from_args(
    args: tuple[Any,...],
) -> xc.DeviceList | None:
  """Returns a representative device list from function call arguments."""
  devices_set: set[xc.DeviceList] = set()
  for x in args:
    sharding = getattr(x, "sharding", None)
    if sharding is not None:
      devices_set.add(x.sharding._internal_device_list)
  if not devices_set:
    return None
  if len(devices_set) != 1:
    raise ValueError(
        f"All arguments must use the same device list, but got {devices_set}."
    )
  return devices_set.pop()


def _compile_to_executable(
    name: str,
    fun: Callable[..., Any],
    in_specs_leaves: tuple[api.ShapeDtypeStruct, ...],
    out_specs_leaves: tuple[api.ShapeDtypeStruct, ...],
    devices: xc.DeviceList,
) -> Callable[..., Any]:
  """Compiles a Python function into a runtime executable."""
  # TODO(hyeontaek): Wrap fun as CustomCallProgram and compile it into an
  # executable.
  del name
  del in_specs_leaves
  del out_specs_leaves
  del devices
  return fun


def _make_output_specs_and_push_result_fun(
    info: FunctionInfo, specialization: Specialization, uid: int
) -> Callable[..., Any]:
  """Creates a function that computes output specs and pushes the result to the result store."""
  assert specialization.in_specs_treedef is not None
  assert specialization.in_specs_leaves is not None
  assert specialization.out_specs_treedef is None
  assert specialization.out_specs_leaves is None
  assert specialization.devices is not None

  devices = specialization.devices

  def lowered_fun(*args, **kwargs) -> Sequence[jax.Array]:
    result = info.fun(*args, **kwargs)
    out_treedef, out_leaves = _tree_flatten(result)
    out_spec_leaves = _get_specs(out_leaves)
    func_backend.SINGLETON_RESULT_STORE.push(uid, out_leaves)
    return serialize_specs(out_treedef, out_spec_leaves, devices)

  _, out_specs_leaves = _tree_flatten(
      make_specs_for_serialized_specs(specialization.devices),
  )
  name = getattr(info.fun, "__name__", "unknown")
  name = f"{name}_output_specs_and_push_result"
  return _compile_to_executable(
      name=name,
      fun=lowered_fun,
      in_specs_leaves=specialization.in_specs_leaves,
      out_specs_leaves=out_specs_leaves,
      devices=specialization.devices,
  )


def _make_pop_result_fun(
    info: FunctionInfo, specialization: Specialization, uid: int
) -> Callable[..., Any]:
  """Makes a function that pops results from the result store."""
  assert specialization.out_specs_treedef is not None
  assert specialization.out_specs_leaves is not None
  assert specialization.devices is not None

  out_specs_treedef = specialization.out_specs_treedef

  def lowered_fun() -> Any:
    flat_result = func_backend.SINGLETON_RESULT_STORE.pop(uid)
    return tree_util.tree_unflatten(out_specs_treedef, flat_result)

  _, in_specs_leaves = _tree_flatten((
      # args
      (),
      # kwargs
      (),
  ))
  name = getattr(info.fun, "__name__", "unknown")
  name = f"{name}_pop_result"
  return _compile_to_executable(
      name=name,
      fun=lowered_fun,
      in_specs_leaves=in_specs_leaves,
      out_specs_leaves=specialization.out_specs_leaves,
      devices=specialization.devices,
  )


def _make_async_execution_fun(
    info: FunctionInfo, specialization: Specialization
) -> Callable[..., Any]:
  """Makes a function that asynchronously executes the function."""
  assert specialization.in_specs_treedef is not None
  assert specialization.in_specs_leaves is not None
  assert specialization.out_specs_treedef is not None
  assert specialization.out_specs_leaves is not None
  assert specialization.devices is not None

  name = getattr(info.fun, "__name__", "unknown")
  return _compile_to_executable(
      name=name,
      fun=info.fun,
      in_specs_leaves=specialization.in_specs_leaves,
      out_specs_leaves=specialization.out_specs_leaves,
      devices=specialization.devices,
  )


@jax.util.cache(max_size=None)
def _get_specialized_func(
    info: FunctionInfo, specialization: Specialization
) -> Callable[..., Any]:
  """Returns a specialized function for the given specialization."""
  assert specialization.in_specs_treedef is not None
  assert specialization.in_specs_leaves is not None
  assert specialization.devices is not None
  uid = random.getrandbits(63)

  mutex = threading.Lock()
  # Asynchronous execution function that has known output_specs.
  async_execution_func = None

  def specialized_func(*args, **kwargs) -> Any:
    """Specialized function to be executed with given args and kwargs."""
    nonlocal specialization, async_execution_func
    with mutex:
      if async_execution_func is None:
        if specialization.out_specs_treedef is None:
          if specialization.out_specs_fn is None:
            serialized_out_specs = _make_output_specs_and_push_result_fun(
                info, specialization, uid
            )(*args, **kwargs)

            # Waits for the output_specs. This may block.
            out_specs_treedef, out_specs_leaves = deserialize_specs(
                serialized_out_specs
            )

            # Subsequent calls would use async_execution_func with discovered
            # output_specs.
            specialization = _apply_specialization(
                specialization,
                out_specs_treedef=out_specs_treedef,
                out_specs_leaves=out_specs_leaves,
            )
            async_execution_func = _make_async_execution_fun(
                info, specialization
            )

            return _make_pop_result_fun(info, specialization, uid)()
          else:
            # Compute out_specs using out_specs_fn and inputs.
            out_specs = specialization.out_specs_fn(*args, **kwargs)
            out_specs_treedef, out_specs_leaves = _tree_flatten(out_specs)
            specialization = _apply_specialization(
                specialization,
                out_specs_treedef=out_specs_treedef,
                out_specs_leaves=out_specs_leaves,
            )
            async_execution_func = _make_async_execution_fun(
                info, specialization
            )
            # Fall-through.
        else:
          async_execution_func = _make_async_execution_fun(info, specialization)
          # Fall-through.

      return async_execution_func(*args, **kwargs)

  return specialized_func


def make_callable(
    fun: Callable[..., Any],
    fun_sourceinfo: str | None,
    fun_signature: inspect.Signature | None,
) -> Callable[..., Any]:
  """Makes a colocated Python callable."""
  return _make_callable(
      FunctionInfo(fun, fun_sourceinfo, fun_signature), Specialization()
  )


def _make_callable(
    info: FunctionInfo,
    specialization: Specialization,
) -> Callable[..., Any]:
  """Internal implementation of make_callable."""

  def specialize(
      in_specs: ShapeDtypeStructTree | None = None,
      out_specs_fn: Callable[..., Any] | None = None,
      out_specs: ShapeDtypeStructTree | None = None,
      devices: Sequence[jax.Device] | None = None,
  ) -> Callable[..., Any]:
    """Returns a colocated Python callable with extra specialization."""
    if in_specs is None:
      in_specs_treedef, in_specs_leaves = None, None
    else:
      in_specs_treedef, in_specs_leaves = _tree_flatten(in_specs)
    if out_specs is None:
      out_specs_treedef, out_specs_leaves = None, None
    else:
      out_specs_treedef, out_specs_leaves = _tree_flatten(out_specs)
    return _make_callable(
        info,
        _apply_specialization(
            specialization,
            in_specs_treedef=in_specs_treedef,
            in_specs_leaves=in_specs_leaves,
            out_specs_fn=out_specs_fn,
            out_specs_treedef=out_specs_treedef,
            out_specs_leaves=out_specs_leaves,
            devices=devices,
        ),
    )

  @api_boundary
  def __call__(*args, **kwargs) -> Any:
    """Executes the function.

    If the output specs are not known, the very first execution will be
    synchronous.
    """
    nonlocal specialization

    in_specs_treedef, args_leaves = _tree_flatten((args, kwargs))

    in_specs_leaves = _get_specs(args_leaves)
    if specialization.in_specs_treedef is None:
      if specialization.out_specs_treedef is None:
        # Allow input polymorphism by applying input_specs specialization
        # temporarily for this call.
        return _make_callable(
            info,
            _apply_specialization(
                specialization,
                in_specs_treedef=in_specs_treedef,
                in_specs_leaves=in_specs_leaves,
            ),
        )(*args, **kwargs)

      # If out_specs is already specialized, we accept only one input_specs
      # permanently by remembering the specialization within this callable
      # itself.
      specialization = _apply_specialization(
          specialization,
          in_specs_treedef=in_specs_treedef,
          in_specs_leaves=in_specs_leaves,
      )
      # Fall-through.

    if specialization.devices is None:
      devices = _infer_devices_from_args(args_leaves)
      if devices is None:
        raise ValueError(
            "No devices found. colocated_python function without input"
            " arguments must be first specialized with devices."
        )
      # Allow device polymorphism by applying devices specialization temporarily
      # for this call.
      return _make_callable(
          info, _apply_specialization(specialization, devices=devices)
      )(*args, **kwargs)

    # To sliences mypy error: Unsupported operand types for != ("PyTreeDef" and
    # "None")  [operator]
    assert isinstance(specialization.in_specs_treedef, tree_util.PyTreeDef)

    # If input_specs is known, verify that it matches actual inputs.
    if (specialization.in_specs_treedef != in_specs_treedef
        or specialization.in_specs_leaves != in_specs_leaves):
      raise ValueError(
          "Input specs do not match: "
          f"Expected ({specialization.in_specs_treedef}, "
          f"{specialization.in_specs_leaves}), "
          f"but got ({in_specs_treedef}, {in_specs_leaves})."
      )

    return _get_specialized_func(info, specialization)(*args, **kwargs)

  __call__ = wraps(info.fun)(__call__)
  __call__.specialize = specialize
  return __call__
