# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Chunk selected graph_trainer module regions for scheduling experiments.

The pass is intentionally placed after CPU offload insertion and before SAC
rematerialization. That keeps offload/reload operating on full activations,
while remat sees and duplicates the chunked compute in backward.

Core terms:
- Region: one concrete forward or backward module root matched from the user
  FQN pattern, such as ``layers.0`` or ``layers.0.moe``.
- Search space: same-direction dependencies inside that root. It is found by
  walking backward from matched region nodes and stopping at placeholders,
  activation-offload ops, other module roots, and saved forward values.
- Live-in: a value used by the selected region but produced outside it.
  Chunkable live-ins have annotated dynamic metadata for the selected mode and
  are split into two equal chunks.
- Live-out: a selected region value consumed outside the region. Full external
  consumers receive a materialized value reconstructed with ``cat`` or, only
  when proven safe, ``add``.
- Provenance live-in: a live-in that is really a per-chunk live-out from an
  earlier planned chunked region. Later chunked regions consume the saved
  per-chunk tuple directly instead of trying to split a potentially
  non-invertible full value, such as MoE token counts.

High-level flow:
1. Import placeholder chunk metadata in ``import_chunk_dim_metadata_pass`` so
   the tracer stays generic.
2. Match concrete module regions from FQN patterns and validate they are nested
   under downstream transformer-block scheduling boundaries.
3. Plan all regions before mutating the graph. Planning discovers search
   spaces, chunkable/provenance live-ins, copied nodes, and live-outs, then
   iterates once more if one chunked region consumes another region's live-out.
4. Transform each plan by inserting two-way splits for chunkable live-ins,
   copying the planned closure once per chunk, and tagging copied nodes with
   ``chunk_id``.
5. Preserve per-chunk live-outs for later chunked consumers. Materialize full
   live-outs only for true non-chunked consumers, with shape validation against
   the original metadata.
6. Erase the original unchunked planned nodes only after every external use has
   been rewired, then lint and recompile the graph.
"""

from __future__ import annotations

import fnmatch
import operator
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

import torch

# Register upstream custom ops used by CPU activation offloading.
import torch._functorch._activation_offloading.offload_ops  # noqa: F401
import torch.fx as fx
from torch._dynamo.graph_deduplication import _stable_topological_sort
from torch.utils._ordered_set import OrderedSet
from torch.utils._pytree import tree_leaves
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.experiments.graph_trainer.common_utils import (
    _dynamic_dim_symbols,
    _earliest_node,
    _free_symbols,
    _get_module_fqn,
    _is_backward_node,
    _is_module_fqn_inside_root,
    _ordered_nodes,
    _tensor_meta,
)
from torchtitan.tools.logging import logger


aten = torch.ops.aten

ChunkMode = Literal["batch", "seq"]
_CHUNK_DIMS_ATTR = "_torchtitan_chunk_dims"
_CHUNK_DIMS_META = "torchtitan_chunk_dims"


_AO_TARGETS = {
    torch.ops.ao.offload.default,
    torch.ops.ao.reload.default,
    torch.ops.ao.wait_tensor.default,
}


@dataclass(frozen=True)
class _Region:
    root_fqn: str
    is_backward: bool
    nodes: tuple[fx.Node, ...]


@dataclass(frozen=True)
class _RegionPlan:
    region: _Region
    search_space: frozenset[fx.Node]
    chunkable_live_ins: frozenset[fx.Node]
    provenance_live_ins: frozenset[fx.Node]
    region_nodes: frozenset[fx.Node]
    region_nodes_tuple: tuple[fx.Node, ...]
    region_live_ins: frozenset[fx.Node]
    live_out_users: dict[fx.Node, tuple[fx.Node, ...]]


def _flatten_module_bucket_plans(plans: list[list[str] | str]) -> tuple[str, ...]:
    roots: list[str] = []
    for plan in plans:
        if isinstance(plan, str):
            roots.append(plan)
        else:
            roots.extend(plan)
    return tuple(roots)


def _transformer_block_roots(
    module_bucket_plans: list[list[str] | str],
) -> tuple[str, ...]:
    return tuple(
        root
        for root in _flatten_module_bucket_plans(module_bucket_plans)
        if root.startswith("layers.")
    )


def _is_excluded_node(node: fx.Node) -> bool:
    """Nodes that must not be duplicated by chunking."""
    if node.op != "call_function":
        return True
    if node.target in _AO_TARGETS:
        return True
    return False


def _is_reverse_closure_boundary(
    node: fx.Node, *, root_fqn: str, is_backward: bool
) -> bool:
    # Step 2.1: reverse closure only discovers a local search space. Boundary
    # nodes may become live-ins, but they must not be duplicated.
    if node.op == "output":
        raise ValueError(
            f"Chunk pass unexpectedly reached graph output while building "
            f"reverse closure for region {root_fqn!r}."
        )
    if node.op == "placeholder":
        return True
    if "chunked_region_fqn" in node.meta:
        return True
    if any(user.op == "output" for user in node.users) and not _fqn(node):
        # Step 2.2: backward closures can reach the graph-owned loss through
        # autograd's ones_like(loss) seed. Keep that scalar outside the region
        # so loss reduction order remains unchanged.
        return True
    if _is_excluded_node(node):
        return True
    fqn = _fqn(node)
    if fqn and not _is_module_fqn_inside_root(fqn, root_fqn):
        return True
    if is_backward and not fqn:
        user_fqns = [_fqn(user) for user in node.users if _fqn(user)]
        if user_fqns and not any(
            _is_module_fqn_inside_root(user_fqn, root_fqn) for user_fqn in user_fqns
        ):
            # Step 2.5: backward graphs contain fqn-less gradient accumulation
            # nodes between module regions. The accumulator belongs to the
            # region that has an immediate module-fqn user; earlier regions see
            # it as an incoming gradient live-in.
            return True
    if (
        is_backward
        and "autograd_backward" not in node.meta
        and any(not _is_backward_node(user) for user in node.users)
    ):
        # Step 2.3: saved forward activations can have module FQN metadata but
        # no backward marker. If they also feed forward users, they are shared
        # boundary values for backward, not backward compute to duplicate.
        return True
    if "autograd_backward" in node.meta and _is_backward_node(node) != is_backward:
        # Step 2.3: backward regions normally consume saved forward activations;
        # those values are live-ins. A forward region depending on backward
        # nodes is structurally invalid for this transform.
        if is_backward:
            return True
        direction = "backward" if is_backward else "forward"
        raise ValueError(
            f"Chunk pass crossed into the opposite graph direction while "
            f"building {direction} reverse closure for region {root_fqn!r}: "
            f"{node.name}."
        )
    return False


def _fqn(node: fx.Node) -> str:
    return _get_module_fqn(node)


def _pattern_root(pattern: str, fqn: str) -> str | None:
    """Return the concrete region root if ``pattern`` matches ``fqn``.

    Patterns are matched segment-by-segment so ``layers.*`` maps
    ``layers.0.attention.wq`` to the concrete root ``layers.0`` instead of
    making one giant region containing every layer.
    """
    pattern_parts = pattern.split(".")
    fqn_parts = fqn.split(".")
    if len(fqn_parts) < len(pattern_parts):
        return None
    for pattern_part, fqn_part in zip(pattern_parts, fqn_parts):
        if not fnmatch.fnmatchcase(fqn_part, pattern_part):
            return None
    return ".".join(fqn_parts[: len(pattern_parts)])


def _reverse_closure_from_boundary_nodes(
    boundary_nodes: list[fx.Node], *, root_fqn: str, is_backward: bool
) -> set[fx.Node]:
    """Collect same-direction dependencies used to discover chunkable sources."""
    closure: set[fx.Node] = set()
    stack = list(boundary_nodes)
    order = {n: i for i, n in enumerate(boundary_nodes[0].graph.nodes)}
    first_boundary_idx = min(order[node] for node in boundary_nodes)

    while stack:
        node = stack.pop()
        if (
            is_backward
            and "autograd_backward" not in node.meta
            and order[node] < first_boundary_idx
        ):
            # Step 2.4: nodes before the first tagged backward node are saved
            # forward values from the original forward pass. Backward chunking
            # can consume them as live-ins/provenance but must not duplicate the
            # same original forward node in the backward region.
            continue
        if node in closure or _is_reverse_closure_boundary(
            node, root_fqn=root_fqn, is_backward=is_backward
        ):
            continue
        closure.add(node)
        stack.extend(node.all_input_nodes)

    return closure


def _forward_closure_from_sources(
    sources: set[fx.Node],
    *,
    search_space: set[fx.Node],
) -> set[fx.Node]:
    """Collect nodes in ``search_space`` that data-depend on chunked sources."""
    closure: set[fx.Node] = set()
    stack = [
        user for source in sources for user in source.users if user in search_space
    ]

    while stack:
        node = stack.pop()
        if node in closure:
            continue
        closure.add(node)
        stack.extend(user for user in node.users if user in search_space)

    return closure


def _is_symbolic_shape_scalar(node: fx.Node) -> bool:
    return _tensor_meta(node) is None and bool(_free_symbols(node.meta.get("val")))


def _find_regions(gm: fx.GraphModule, patterns: list[str]) -> list[_Region]:
    matched_boundaries: dict[tuple[str, bool], list[fx.Node]] = defaultdict(list)

    for node in gm.graph.nodes:
        if _is_excluded_node(node):
            continue
        fqn = _fqn(node)
        if not fqn:
            continue
        roots = [root for p in patterns if (root := _pattern_root(p, fqn))]
        if not roots:
            continue
        if len(set(roots)) > 1:
            raise ValueError(
                f"Chunk pass patterns match node {node.name!r} ambiguously: {roots}"
            )
        root = roots[0]
        # Step 1: FQN matching identifies user-selected anchors only. The copied
        # region is computed later from activation live-ins, which avoids
        # duplicating parameter-only prep such as weight all-gathers or casts.
        matched_boundaries[(root, _is_backward_node(node))].append(node)

    order = {n: i for i, n in enumerate(gm.graph.nodes)}
    regions = [
        _Region(root, is_backward, tuple(sorted(boundary_nodes, key=order.__getitem__)))
        for (root, is_backward), boundary_nodes in matched_boundaries.items()
    ]
    return sorted(regions, key=lambda r: min(order[n] for n in r.nodes))


def _validate_regions_within_boundaries(
    regions: list[_Region],
    *,
    module_bucket_plans: list[list[str] | str] | None,
) -> None:
    if module_bucket_plans is None:
        return

    boundary_roots = _transformer_block_roots(module_bucket_plans)
    if not boundary_roots:
        return

    invalid_roots = sorted(
        {
            region.root_fqn
            for region in regions
            if not any(
                _is_module_fqn_inside_root(region.root_fqn, boundary)
                for boundary in boundary_roots
            )
        }
    )
    if invalid_roots:
        raise ValueError(
            "Chunk module roots must be equal to or nested under downstream "
            "transformer-block scheduling boundaries. Invalid roots: "
            f"{invalid_roots}; boundaries: {list(boundary_roots)}."
        )


def _annotated_chunk_dim(node: fx.Node, mode: ChunkMode) -> int | None:
    chunk_dims = node.meta.get(_CHUNK_DIMS_META)
    if not isinstance(chunk_dims, dict):
        return None
    spec = chunk_dims.get(mode)
    if isinstance(spec, dict) and isinstance(spec.get("dim"), int):
        return spec["dim"]
    return None


def _annotated_chunk_hint(node: fx.Node, mode: ChunkMode) -> int | None:
    chunk_dims = node.meta.get(_CHUNK_DIMS_META)
    if not isinstance(chunk_dims, dict):
        return None
    spec = chunk_dims.get(mode)
    if isinstance(spec, dict) and isinstance(spec.get("hint"), int):
        return spec["hint"]
    return None


def _range_upper_bound(value: object) -> int | None:
    sym_node = getattr(value, "node", None)
    shape_env = getattr(sym_node, "shape_env", None)
    expr = getattr(sym_node, "expr", None)
    if shape_env is None or expr is None:
        return None
    value_range = getattr(shape_env, "var_to_range", {}).get(expr)
    upper = getattr(value_range, "upper", None)
    try:
        return int(upper) if upper is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def _dynamic_dims_with_hints(val: torch.Tensor) -> list[tuple[int, int]]:
    dynamic_dims: list[tuple[int, int]] = []
    for dim, size in enumerate(val.shape):
        if not _free_symbols(size):
            continue
        hint = _range_upper_bound(size)
        if hint is None:
            raise ValueError(
                f"Chunk pass could not infer a hint for dynamic dimension {dim} "
                f"of shape {tuple(val.shape)}."
            )
        dynamic_dims.append((dim, hint))
    return dynamic_dims


def import_chunk_dim_metadata_pass(
    gm: fx.GraphModule,
    example_inputs: tuple[Any, ...] | None = None,
    *,
    mode: ChunkMode,
) -> fx.GraphModule:
    """Import chunk dimension metadata onto placeholders before chunk planning.

    The tracer stays pass-agnostic: this pass reads explicit TorchTitan metadata
    when present, otherwise it derives the selected chunk dimension from the
    single dynamic placeholder dimension created by ``mark_chunk_dynamic_dims``.
    """
    if mode not in ("batch", "seq"):
        raise ValueError(f"Unknown chunk mode: {mode!r}")

    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    imported = 0
    for idx, node in enumerate(placeholders):
        chunk_dims = dict(node.meta.get(_CHUNK_DIMS_META, {}))
        if example_inputs is not None and idx < len(example_inputs):
            explicit = getattr(example_inputs[idx], _CHUNK_DIMS_ATTR, None)
            if explicit is not None:
                chunk_dims.update(dict(explicit))

        if mode not in chunk_dims:
            val = _tensor_meta(node)
            if val is None:
                continue
            dynamic_dims = _dynamic_dims_with_hints(val)
            if not dynamic_dims:
                continue
            if len(dynamic_dims) != 1:
                raise ValueError(
                    f"Chunk pass expected one dynamic placeholder dimension for "
                    f"{node.name}, found {dynamic_dims}."
                )
            dim, hint = dynamic_dims[0]
            chunk_dims[mode] = {"dim": dim, "hint": hint}

        node.meta[_CHUNK_DIMS_META] = chunk_dims
        imported += 1

    logger.debug(
        "Imported chunk_%s placeholder metadata for %d placeholders",
        mode,
        imported,
    )
    return gm


def _static_placeholders(gm: fx.GraphModule, num_static_inputs: int) -> set[fx.Node]:
    return {
        node
        for idx, node in enumerate(n for n in gm.graph.nodes if n.op == "placeholder")
        if idx < num_static_inputs
    }


def _static_derived_nodes(
    gm: fx.GraphModule, static_placeholders: set[fx.Node]
) -> set[fx.Node]:
    static_nodes = set(static_placeholders)
    for node in gm.graph.nodes:
        if node in static_nodes:
            continue
        inputs = node.all_input_nodes
        if inputs and all(inp in static_nodes for inp in inputs):
            static_nodes.add(node)
    return static_nodes


def _chunk_size(val: torch.Tensor, dim: int) -> int | torch.SymInt:
    size = val.shape[dim]
    if isinstance(size, torch.SymInt):
        return size // 2
    size = int(size)
    if size % 2 != 0:
        raise ValueError(
            f"Cannot split dimension {dim} of shape {tuple(val.shape)} into two "
            "equal chunks."
        )
    return size // 2


def _expr(value: object) -> object:
    return getattr(getattr(value, "node", None), "expr", value)


def _expr_matches(lhs: object, rhs: object) -> bool:
    lhs_expr = _expr(lhs)
    rhs_expr = _expr(rhs)
    if lhs_expr == rhs_expr:
        return True
    try:
        return bool((lhs_expr - rhs_expr) == 0)
    except (TypeError, ValueError):
        return False


def _chunk_size_for_live_in(
    node: fx.Node,
    mode: ChunkMode,
    val: torch.Tensor,
    dim: int,
    *,
    dim_chunk_sizes: list[tuple[object, object]] | None = None,
) -> int | torch.SymInt:
    hint = _annotated_chunk_hint(node, mode)
    if hint is not None:
        if hint % 2 != 0:
            raise ValueError(
                f"Cannot split annotated {mode} dimension with hint {hint} "
                "into two equal chunks."
            )
    full_dim = val.shape[dim]
    for candidate_full_dim, candidate_chunk_dim in dim_chunk_sizes or ():
        if _expr_matches(full_dim, candidate_full_dim):
            return candidate_chunk_dim
    return _chunk_size(val, dim)


def _chunk_meta(
    val: torch.Tensor, dim: int, chunk_size: int | torch.SymInt
) -> torch.Tensor:
    shape = list(val.shape)
    shape[dim] = chunk_size
    return val.new_empty(shape)


def _propagate_chunk_dim_metadata(gm: fx.GraphModule) -> None:
    symbol_specs: dict[object, dict[ChunkMode, int]] = defaultdict(dict)
    for node in gm.graph.nodes:
        val = _tensor_meta(node)
        if val is None:
            continue
        for mode in ("batch", "seq"):
            dim = _annotated_chunk_dim(node, mode)
            hint = _annotated_chunk_hint(node, mode)
            if dim is None or hint is None:
                continue
            for symbol in _dynamic_dim_symbols(val, dim):
                if _expr_matches(val.shape[dim], symbol):
                    symbol_specs[symbol][mode] = hint

    if not symbol_specs:
        return

    for node in gm.graph.nodes:
        val = _tensor_meta(node)
        if val is None:
            continue
        inferred: dict[ChunkMode, tuple[int, int] | None] = {}
        for dim in range(val.dim()):
            for symbol in _dynamic_dim_symbols(val, dim):
                for mode, hint in symbol_specs.get(symbol, {}).items():
                    spec = inferred.get(mode)
                    if spec is None and mode in inferred:
                        continue
                    dim_hint = _evaluate_sympy_with_hints(
                        val.shape[dim], {symbol: hint}
                    )
                    if dim_hint is None:
                        continue
                    new_spec = (dim, dim_hint)
                    inferred[mode] = new_spec if spec in (None, new_spec) else None

        chunk_dims = dict(node.meta.get(_CHUNK_DIMS_META, {}))
        for mode, spec in inferred.items():
            if spec is None:
                continue
            dim, hint = spec
            existing = chunk_dims.get(mode)
            if isinstance(existing, dict):
                if existing.get("dim") != dim:
                    continue
            else:
                chunk_dims[mode] = {"dim": dim, "hint": hint}
        if chunk_dims:
            node.meta[_CHUNK_DIMS_META] = chunk_dims


def _evaluate_sympy_with_hints(
    expr: object, symbol_full_sizes: dict[object, int]
) -> int | None:
    try:
        symbols = _free_symbols(expr)
        if any(symbol not in symbol_full_sizes for symbol in symbols):
            return None
        expr = _expr(expr)
        expr = expr.subs({symbol: symbol_full_sizes[symbol] for symbol in symbols})
        if _free_symbols(expr):
            return None
        return int(expr)
    except (AttributeError, TypeError, ValueError):
        return None


def _collect_symbol_full_sizes(
    gm: fx.GraphModule, mode: ChunkMode
) -> dict[object, int]:
    symbol_full_sizes: dict[object, int] = {}
    for node in gm.graph.nodes:
        val = _tensor_meta(node)
        if val is None:
            continue
        dim = _annotated_chunk_dim(node, mode)
        hint = _annotated_chunk_hint(node, mode)
        if dim is None or hint is None:
            continue
        for symbol in _dynamic_dim_symbols(val, dim):
            if _expr_matches(val.shape[dim], symbol):
                existing = symbol_full_sizes.get(symbol)
                if existing is not None and existing != hint:
                    raise ValueError(
                        f"Chunk pass found conflicting hints for symbol {symbol}: "
                        f"{existing} and {hint}."
                    )
                symbol_full_sizes[symbol] = hint
    return symbol_full_sizes


def _make_symbol_half(
    symbol: object, *, like_value: object, symbol_full_sizes: dict[object, int]
) -> object:
    shape_env = getattr(getattr(like_value, "node", None), "shape_env", None)
    chunk_expr = symbol // 2
    hint = symbol_full_sizes[symbol] // 2
    if shape_env is None:
        return chunk_expr
    return shape_env.create_symintnode(chunk_expr, hint=hint)


def _replace_chunk_symbols(
    value: object,
    *,
    dim_chunk_sizes: list[tuple[object, object]],
    symbol_chunk_values: dict[object, object],
    symbol_full_sizes: dict[object, int],
) -> object:
    for full_dim, chunk_dim in dim_chunk_sizes:
        if _expr_matches(value, full_dim):
            return chunk_dim

    symbols = _free_symbols(value)
    if not (symbols & set(symbol_chunk_values)):
        return value

    expr = _expr(value)
    shape_env = getattr(getattr(value, "node", None), "shape_env", None)
    replacements = {
        symbol: _expr(symbol_chunk_values[symbol])
        for symbol in symbols
        if symbol in symbol_chunk_values
    }
    try:
        chunk_expr = expr.subs(replacements)
    except AttributeError:
        return value

    if shape_env is None:
        return _evaluate_sympy_with_hints(chunk_expr, symbol_full_sizes) or value

    hint = _evaluate_sympy_with_hints(chunk_expr, symbol_full_sizes)
    return shape_env.create_symintnode(chunk_expr, hint=hint)


def _chunked_meta_from_original(
    original: fx.Node,
    *,
    dim_chunk_sizes: list[tuple[object, object]],
    symbol_chunk_values: dict[object, object],
    symbol_full_sizes: dict[object, int],
) -> torch.Tensor | None:
    val = _tensor_meta(original)
    if val is None:
        return None
    shape = []
    changed = False
    for dim in val.shape:
        chunk_dim = _replace_chunk_symbols(
            dim,
            dim_chunk_sizes=dim_chunk_sizes,
            symbol_chunk_values=symbol_chunk_values,
            symbol_full_sizes=symbol_full_sizes,
        )
        changed = changed or chunk_dim is not dim
        shape.append(chunk_dim)
    if not changed:
        return val
    return val.new_empty(shape)


def _chunk_dim_for_live_in(node: fx.Node, mode: ChunkMode) -> int:
    val = _tensor_meta(node)
    if val is None:
        raise ValueError(f"Chunk live-in {node.name} has no tensor metadata.")
    annotated_dim = _annotated_chunk_dim(node, mode)
    if annotated_dim is not None:
        return annotated_dim

    raise ValueError(
        f"Chunk pass requires an annotated {mode} dimension for live-in "
        f"{node.name} with shape {tuple(val.shape)}."
    )


def _chunk_symbol_dims(node: fx.Node, chunk_symbols: frozenset[object]) -> list[int]:
    val = _tensor_meta(node)
    if val is None:
        return []
    return [
        dim
        for dim in range(val.dim())
        if _dynamic_dim_symbols(val, dim) & chunk_symbols
    ]


def _combine_kind_and_dim(
    node: fx.Node,
    mode: ChunkMode,
    chunk_symbols: frozenset[object],
) -> tuple[Literal["cat", "add"], int | None]:
    matching_dims = _chunk_symbol_dims(node, chunk_symbols)
    if len(matching_dims) == 1:
        return "cat", matching_dims[0]
    if not matching_dims:
        annotated_dim = _annotated_chunk_dim(node, mode)
        if annotated_dim is not None:
            return "cat", annotated_dim
        return "add", None
    raise ValueError(
        f"Chunk pass found multiple chunk-symbol dims for live-out {node.name}: "
        f"{matching_dims}"
    )


def mark_chunk_dynamic_dims(
    tensor: torch.Tensor,
    *,
    mode: ChunkMode,
) -> None:
    """Mark graph_trainer's main input dimensions used by chunk passes.

    ``minimal_fx_tracer`` currently supports ``mark_unbacked`` rather than
    ``mark_dynamic``. We still attach explicit TorchTitan metadata so the graph
    pass can distinguish batch and sequence dims without relying on rank.
    """
    from torch._dynamo.decorators import mark_unbacked

    dims: dict[str, dict[str, int]] = dict(getattr(tensor, _CHUNK_DIMS_ATTR, {}))

    def mark(dim: int) -> None:
        if tensor.dim() <= dim:
            raise ValueError(
                f"Cannot mark {mode} chunk dim {dim} for input shape "
                f"{tuple(tensor.shape)}."
            )
        hint = int(tensor.shape[dim])
        mark_unbacked(
            tensor,
            dim,
            hint_override=hint,
            min=2,
            max=hint,
            shape_id=f"torchtitan_chunk_{mode}",
        )
        dims[mode] = {"dim": dim, "hint": hint}

    if mode == "batch":
        mark(0)
    elif mode == "seq":
        mark(1)
    else:
        raise ValueError(f"Unknown chunk mode: {mode!r}")

    setattr(tensor, _CHUNK_DIMS_ATTR, dims)


def prepare_ep_overlap_trace_inputs(
    compile_config: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    """Apply EP-overlap input annotations before make_fx tracing.

    ``trace_train_step`` calls this hook immediately before tracing so
    overlap-specific trace preparation stays close to the internal chunking
    implementation. The graph pass needs these annotations before tracing
    because make_fx must see a symbolic batch/sequence dimension.
    """
    if "ep_overlap" not in getattr(compile_config, "passes", []):
        return
    if not compile_config.ep_overlap_modules:
        raise ValueError(
            "--compile.ep_overlap_modules must be non-empty when "
            "--compile.passes contains ep_overlap"
        )
    if not args or not isinstance(args[0], torch.Tensor):
        raise ValueError(
            "ep_overlap tracing expects the first user input to be a Tensor"
        )
    mode = compile_config.ep_overlap_mode
    if mode == "batch":
        dim = 0
    elif mode == "seq":
        dim = 1
    else:
        raise ValueError(f"Unknown ep_overlap_mode: {mode!r}")
    main_input = args[0]
    hint = int(main_input.shape[dim])

    def mark_if_matching(value: object) -> None:
        if (
            isinstance(value, torch.Tensor)
            and value.dim() > dim
            and int(value.shape[dim]) == hint
        ):
            mark_chunk_dynamic_dims(value, mode=mode)

    for value in [*tree_leaves(args), *tree_leaves(kwargs)]:
        mark_if_matching(value)


def _is_chunkable_live_in(
    node: fx.Node,
    *,
    mode: ChunkMode,
    static_nodes: set[fx.Node],
) -> bool:
    if node in static_nodes:
        return False
    val = _tensor_meta(node)
    if val is None:
        return False
    # Step 7: missing chunk metadata means this boundary value is not an activation
    # source for the selected mode. Invalid annotated sizes should still raise
    # so user configuration mistakes do not become silent no-ops.
    try:
        dim = _chunk_dim_for_live_in(node, mode)
    except ValueError:
        return False
    _chunk_size_for_live_in(node, mode, val, dim)
    return True


def _copy_meta(src: fx.Node, dst: fx.Node, *, chunk_id: int | None = None) -> None:
    # Metadata contains FakeTensors and other objects that should not be deep-copied.
    dst.meta = dict(src.meta)
    if "custom" in dst.meta:
        dst.meta["custom"] = dict(dst.meta["custom"])
    if chunk_id is not None:
        dst.meta["chunk_id"] = chunk_id


def _rename(node: fx.Node, candidate: str) -> None:
    node._rename(candidate)


def _set_recompute_like(src: fx.Node, dst: fx.Node) -> None:
    if "recompute" in src.meta:
        dst.meta["recompute"] = src.meta["recompute"]
    if src.meta.get("recompute") is CheckpointPolicy.MUST_CPU_OFFLOAD:
        # CPU offload was already inserted before chunking. The new full-value
        # cat feeds that existing offload op; it must not be interpreted as a
        # fresh offload candidate by later passes.
        dst.meta["recompute"] = CheckpointPolicy.MUST_CPU_OFFLOAD


def _infer_node_val(node: fx.Node) -> None:
    def meta_arg(arg: object) -> object:
        if isinstance(arg, fx.Node) and "val" in arg.meta:
            return arg.meta["val"]
        return arg

    try:
        args = torch.fx.map_arg(node.args, meta_arg)
        kwargs = torch.fx.map_arg(node.kwargs, meta_arg)
        node.meta["val"] = node.target(*args, **kwargs)
    except Exception:
        pass


def _dim_matches(
    original_dim: object,
    materialized_dim: object,
    symbol_full_sizes: dict[object, int],
) -> bool:
    original_symbols = _free_symbols(original_dim)
    if original_symbols:
        original_size = _evaluate_with_hints(original_dim, symbol_full_sizes)
        materialized_size = _evaluate_with_hints(materialized_dim, symbol_full_sizes)
        if original_size is not None and materialized_size is not None:
            return original_size == materialized_size
        # Unknown symbolic dims are intentionally left symbolic; avoid forcing a
        # guard just to validate metadata.
        return any(symbol not in symbol_full_sizes for symbol in original_symbols)

    if _free_symbols(materialized_dim):
        original_size = _evaluate_with_hints(original_dim, symbol_full_sizes)
        materialized_size = _evaluate_with_hints(materialized_dim, symbol_full_sizes)
        if original_size is not None and materialized_size is not None:
            return original_size == materialized_size
        return True

    return int(original_dim) == int(materialized_dim)


def _evaluate_with_hints(
    value: object, symbol_full_sizes: dict[object, int]
) -> int | None:
    symbols = _free_symbols(value)
    if not symbols:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    if any(symbol not in symbol_full_sizes for symbol in symbols):
        return None

    expr = getattr(getattr(value, "node", None), "expr", value)
    try:
        expr = expr.subs({symbol: symbol_full_sizes[symbol] for symbol in symbols})
    except AttributeError:
        return None
    if _free_symbols(expr):
        return None
    try:
        return int(expr)
    except (TypeError, ValueError):
        return None


def _set_materialized_chunk_dim_meta(
    materialized: fx.Node,
    original: fx.Node,
    *,
    mode: ChunkMode,
    dim: int,
    symbol_full_sizes: dict[object, int],
) -> None:
    chunk_dims = dict(original.meta.get(_CHUNK_DIMS_META, {}))
    hint = _annotated_chunk_hint(original, mode)
    if hint is None:
        val = _tensor_meta(original)
        if val is not None:
            hint = _evaluate_with_hints(val.shape[dim], symbol_full_sizes)
    if hint is not None:
        chunk_dims[mode] = {"dim": dim, "hint": hint}
        materialized.meta[_CHUNK_DIMS_META] = chunk_dims


def _validate_materialized_meta(
    materialized: fx.Node,
    original: fx.Node,
    symbol_full_sizes: dict[object, int],
) -> None:
    original_val = _tensor_meta(original)
    materialized_val = _tensor_meta(materialized)
    if not isinstance(original_val, torch.Tensor) or not isinstance(
        materialized_val, torch.Tensor
    ):
        return
    if len(original_val.shape) != len(materialized_val.shape):
        raise RuntimeError(
            f"Chunk pass materialized {materialized.name} with rank "
            f"{len(materialized_val.shape)}, expected {len(original_val.shape)} "
            f"from original {original.name}."
        )
    if not all(
        _dim_matches(original_dim, materialized_dim, symbol_full_sizes)
        for original_dim, materialized_dim in zip(
            original_val.shape, materialized_val.shape
        )
    ):
        raise RuntimeError(
            f"Chunk pass materialized {materialized.name} with shape "
            f"{tuple(materialized_val.shape)}, expected {tuple(original_val.shape)} "
            f"from original {original.name}."
        )


def _map_arg_for_chunk(
    arg: object,
    *,
    copied: dict[fx.Node, fx.Node],
    split_live_ins: dict[fx.Node, tuple[fx.Node, fx.Node]],
    chunk_id: int,
) -> object:
    if isinstance(arg, fx.Node):
        if arg in copied:
            return copied[arg]
        chunks = split_live_ins.get(arg)
        if chunks is not None:
            return chunks[chunk_id]
    return arg


def _insert_symbolic_split_size(
    gm: fx.GraphModule,
    live_in: fx.Node,
    *,
    dim: int,
    meta_chunk_size: int | torch.SymInt,
) -> fx.Node:
    full_size = gm.graph.call_function(aten.sym_size.int, args=(live_in, dim))
    full_size.meta["val"] = _tensor_meta(live_in).shape[dim]

    remainder = gm.graph.call_function(operator.mod, args=(full_size, 2))
    is_even = gm.graph.call_function(operator.eq, args=(remainder, 0))
    gm.graph.call_function(
        aten._assert_scalar.default,
        args=(is_even, f"chunk pass requires dimension {dim} to be even"),
    )

    half_size = gm.graph.call_function(operator.floordiv, args=(full_size, 2))
    half_size.meta["val"] = meta_chunk_size
    return half_size


def _insert_symbolic_half_scalar(
    gm: fx.GraphModule,
    scalar: fx.Node,
    *,
    meta_half_size: object,
) -> fx.Node:
    remainder = gm.graph.call_function(operator.mod, args=(scalar, 2))
    is_even = gm.graph.call_function(operator.eq, args=(remainder, 0))
    gm.graph.call_function(
        aten._assert_scalar.default,
        args=(is_even, "chunk pass requires symbolic scalar live-in to be even"),
    )

    half_size = gm.graph.call_function(operator.floordiv, args=(scalar, 2))
    half_size.meta["val"] = meta_half_size
    return half_size


def _safe_erase_region(region_nodes: tuple[fx.Node, ...]) -> None:
    graph = region_nodes[0].graph if region_nodes else None
    for node in reversed(region_nodes):
        if node.users:
            raise RuntimeError(
                f"Chunk pass could not erase original node {node.name}; "
                f"remaining users: {[u.name for u in node.users]}"
            )
        node.graph.erase_node(node)
    if graph is not None:
        remaining = [node.name for node in region_nodes if node in graph.nodes]
        if remaining:
            raise RuntimeError(
                "Chunk pass failed to erase original region nodes: " f"{remaining}"
            )


def _earliest_region_user(
    live_in: fx.Node, region_nodes: tuple[fx.Node, ...], order: dict[fx.Node, int]
) -> fx.Node:
    users = [node for node in region_nodes if live_in in node.all_input_nodes]
    return _earliest_node(users, order)


def _constant_factor(numerator: object, denominator: object) -> int | None:
    try:
        factor = _expr(numerator) / _expr(denominator)
        if _free_symbols(factor):
            return None
        if int(factor) == factor:
            return int(factor)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return None


def _scalar_matches_split_dim(
    scalar_val: object,
    *,
    mode: ChunkMode,
    split_live_ins: dict[fx.Node, tuple[fx.Node, fx.Node]],
) -> bool:
    for live_in in split_live_ins:
        val = _tensor_meta(live_in)
        if val is None:
            continue
        dim = _annotated_chunk_dim(live_in, mode)
        if dim is None:
            continue
        if (
            _constant_factor(scalar_val, val.shape[dim]) is not None
            or _constant_factor(val.shape[dim], scalar_val) is not None
        ):
            return True
    return False


def _scalar_sum_matches(
    chunk0_val: object, chunk1_val: object, original_val: object
) -> bool:
    try:
        return _expr_matches(_expr(chunk0_val) + _expr(chunk1_val), original_val)
    except (TypeError, ValueError):
        return False


def _scalar_halves_match_guarded_even_original(
    chunk0_val: object, chunk1_val: object, original_val: object
) -> bool:
    try:
        half_original = _expr(original_val) // 2
        return _expr_matches(chunk0_val, half_original) and _expr_matches(
            chunk1_val, half_original
        )
    except (TypeError, ValueError):
        return False


def _materialize_scalar_live_out(
    gm: fx.GraphModule,
    live_out: fx.Node,
    copied0: fx.Node,
    copied1: fx.Node,
    *,
    mode: ChunkMode,
    split_live_ins: dict[fx.Node, tuple[fx.Node, fx.Node]],
) -> fx.Node:
    original_val = live_out.meta.get("val")
    chunk0_val = copied0.meta.get("val")
    chunk1_val = copied1.meta.get("val")

    if _expr_matches(chunk0_val, original_val) and _expr_matches(
        chunk1_val, original_val
    ):
        return copied0

    if _scalar_sum_matches(chunk0_val, chunk1_val, original_val):
        materialized = gm.graph.call_function(operator.add, args=(copied0, copied1))
        _rename(materialized, f"{live_out.name}_chunk_sum")
        materialized.meta["val"] = original_val
        return materialized

    if _scalar_halves_match_guarded_even_original(
        chunk0_val, chunk1_val, original_val
    ) or (
        _expr_matches(chunk0_val, chunk1_val)
        and _scalar_matches_split_dim(
            original_val,
            mode=mode,
            split_live_ins=split_live_ins,
        )
    ):
        # Step 9.3: split insertion asserted the symbolic dimension is even.
        # Under that guard, two copied sym_size halves reconstruct the original
        # full symbolic size even when general SymPy simplification cannot
        # prove floor(u0 / 2) + floor(u0 / 2) == u0 for an unconstrained symbol.
        materialized = gm.graph.call_function(operator.add, args=(copied0, copied1))
        _rename(materialized, f"{live_out.name}_chunk_sum")
        materialized.meta["val"] = original_val
        return materialized

    raise ValueError(
        f"Chunk pass cannot materialize scalar live-out {live_out.name}: "
        f"chunk values {chunk0_val!r}, {chunk1_val!r} do not reconstruct "
        f"original value {original_val!r}."
    )


def _region_contains_attention(region: _Region) -> bool:
    return any("attention" in _fqn(n).split(".") for n in region.nodes)


def _candidate_live_ins(search_space: set[fx.Node]) -> set[fx.Node]:
    return {
        inp
        for node in search_space
        for inp in node.all_input_nodes
        if inp not in search_space
    }


def _chunkable_live_ins(
    live_ins: set[fx.Node],
    *,
    mode: ChunkMode,
    static_nodes: set[fx.Node],
) -> set[fx.Node]:
    return {
        live_in
        for live_in in live_ins
        if _is_chunkable_live_in(
            live_in,
            mode=mode,
            static_nodes=static_nodes,
        )
    }


def _make_region_plan(
    gm: fx.GraphModule,
    region: _Region,
    *,
    mode: ChunkMode,
    static_nodes: set[fx.Node],
    provenance_sources: set[fx.Node],
) -> _RegionPlan:
    if mode == "seq" and _region_contains_attention(region):
        raise NotImplementedError(
            f"chunk_seq for attention-containing region {region.root_fqn!r} "
            "requires a full-K/V attention rewrite and is not implemented in v1."
        )

    order = _ordered_nodes(gm)
    search_space = _reverse_closure_from_boundary_nodes(
        list(region.nodes),
        root_fqn=region.root_fqn,
        is_backward=region.is_backward,
    )
    if not search_space:
        raise ValueError(
            f"Chunk pass matched region {region.root_fqn!r}, but found no "
            "same-direction nodes inside that module root."
        )

    candidate_live_ins = _candidate_live_ins(search_space)
    chunkable_live_ins = _chunkable_live_ins(
        candidate_live_ins,
        mode=mode,
        static_nodes=static_nodes,
    )
    provenance_live_ins = candidate_live_ins & provenance_sources

    # Compute the copied descendant closure to a local fixed point. Backward
    # regions can expose saved forward activations or planned provenance values
    # only after the first activation-dependent slice has been selected.
    while True:
        region_nodes = _forward_closure_from_sources(
            chunkable_live_ins | provenance_live_ins,
            search_space=search_space,
        )
        region_live_ins = {
            inp
            for node in region_nodes
            for inp in node.all_input_nodes
            if inp not in region_nodes
        }
        extra_chunkable_live_ins = _chunkable_live_ins(
            region_live_ins - chunkable_live_ins,
            mode=mode,
            static_nodes=static_nodes,
        )
        extra_provenance_live_ins = (
            region_live_ins & provenance_sources
        ) - provenance_live_ins
        if not extra_chunkable_live_ins and not extra_provenance_live_ins:
            break
        chunkable_live_ins |= extra_chunkable_live_ins
        provenance_live_ins |= extra_provenance_live_ins

    if not region_nodes:
        raise ValueError(
            f"Chunk pass matched region {region.root_fqn!r}, but found no "
            f"activation-dependent nodes for chunk_{mode}."
        )

    region_nodes_tuple = tuple(sorted(region_nodes, key=order.__getitem__))
    region_live_ins = {
        inp
        for node in region_nodes_tuple
        for inp in node.all_input_nodes
        if inp not in region_nodes
    }
    chunkable_live_ins = (chunkable_live_ins & region_live_ins) - region_nodes
    provenance_live_ins = (provenance_live_ins & region_live_ins) - region_nodes
    live_outs = [
        node
        for node in region_nodes_tuple
        if any(user not in region_nodes for user in node.users)
    ]
    if not live_outs:
        raise ValueError(
            f"Chunk pass found no live-outs for activation-dependent region "
            f"{region.root_fqn!r}."
        )

    return _RegionPlan(
        region=region,
        search_space=frozenset(search_space),
        chunkable_live_ins=frozenset(chunkable_live_ins),
        provenance_live_ins=frozenset(provenance_live_ins),
        region_nodes=frozenset(region_nodes),
        region_nodes_tuple=region_nodes_tuple,
        region_live_ins=frozenset(region_live_ins),
        live_out_users={
            live_out: tuple(user for user in live_out.users if user not in region_nodes)
            for live_out in live_outs
        },
    )


def _plan_regions(
    gm: fx.GraphModule,
    regions: list[_Region],
    *,
    mode: ChunkMode,
    static_nodes: set[fx.Node],
) -> list[_RegionPlan]:
    provenance_sources: set[fx.Node] = set()
    plans: list[_RegionPlan] = []

    while True:
        plans = [
            _make_region_plan(
                gm,
                region,
                mode=mode,
                static_nodes=static_nodes,
                provenance_sources=provenance_sources,
            )
            for region in regions
        ]
        chunk_consumer_nodes = _chunk_consumer_nodes(plans)
        next_provenance_sources = {
            live_out
            for plan in plans
            for live_out, users in plan.live_out_users.items()
            if any(user in chunk_consumer_nodes for user in users)
        }
        if next_provenance_sources == provenance_sources:
            break
        provenance_sources = next_provenance_sources

    _validate_disjoint_plans(plans)
    order = _ordered_nodes(gm)
    return sorted(
        plans,
        key=lambda plan: min(order[node] for node in plan.region_nodes_tuple),
    )


def _chunk_consumer_nodes(plans: list[_RegionPlan]) -> set[fx.Node]:
    return {node for plan in plans for node in plan.region_nodes}


def _validate_disjoint_plans(plans: list[_RegionPlan]) -> None:
    owner: dict[fx.Node, _RegionPlan] = {}
    for plan in plans:
        for node in plan.region_nodes:
            previous = owner.get(node)
            if previous is not None:
                if _is_symbolic_shape_scalar(node):
                    # Step 3.2: symbolic shape helpers such as sym_size nodes can
                    # be shared by adjacent module regions. They are pure shape
                    # compute, and each region copy rewrites them against that
                    # region's chunked live-ins.
                    continue
                raise ValueError(
                    "Chunk pass planned overlapping regions for node "
                    f"{node.name}: {previous.region.root_fqn!r} "
                    f"({'backward' if previous.region.is_backward else 'forward'}) "
                    f"and {plan.region.root_fqn!r} "
                    f"({'backward' if plan.region.is_backward else 'forward'})."
                )
            owner[node] = plan


def _is_additive_partial_sum_passthrough(node: fx.Node) -> bool:
    if node.op != "call_function":
        return False
    return node.target in (operator.add, aten.add.Tensor, aten.cumsum.default)


def _is_partial_sum_seed(node: fx.Node, chunk_symbols: frozenset[object]) -> bool:
    if node.op != "call_function":
        return False
    if "histc" in str(node.target) or "bincount" in str(node.target):
        return True
    if node.target in (aten.sum.default, aten.sum.dim_IntList):
        for input_node in node.all_input_nodes:
            val = _tensor_meta(input_node)
            if val is not None and any(
                _dynamic_dim_symbols(val, dim) & chunk_symbols
                for dim in range(val.dim())
            ):
                return True
    return False


def _is_partial_sum_value(
    node: fx.Node,
    *,
    plan_nodes: frozenset[fx.Node],
    chunk_symbols: frozenset[object],
    memo: dict[fx.Node, bool],
) -> bool:
    cached = memo.get(node)
    if cached is not None:
        return cached
    if node not in plan_nodes:
        memo[node] = False
        return False
    if _is_partial_sum_seed(node, chunk_symbols):
        memo[node] = True
        return True
    if not _is_additive_partial_sum_passthrough(node):
        memo[node] = False
        return False
    input_nodes = node.all_input_nodes
    result = bool(input_nodes) and any(
        _is_partial_sum_value(
            input_node,
            plan_nodes=plan_nodes,
            chunk_symbols=chunk_symbols,
            memo=memo,
        )
        for input_node in input_nodes
    )
    memo[node] = result
    return result


def _is_graph_output_value(node: fx.Node) -> bool:
    return any(user.op == "output" for user in node.users)


def _can_materialize_add(
    live_out: fx.Node,
    *,
    region: _Region,
    plan_nodes: frozenset[fx.Node],
    chunk_symbols: frozenset[object],
    partial_sum_memo: dict[fx.Node, bool],
) -> bool:
    if region.is_backward:
        return _is_graph_output_value(live_out)
    return _is_partial_sum_value(
        live_out,
        plan_nodes=plan_nodes,
        chunk_symbols=chunk_symbols,
        memo=partial_sum_memo,
    )


def _transform_region(
    gm: fx.GraphModule,
    plan: _RegionPlan,
    *,
    mode: ChunkMode,
    all_chunk_consumer_nodes: set[fx.Node],
    chunk_value_nodes: dict[fx.Node, tuple[fx.Node, fx.Node]],
) -> int:
    order = _ordered_nodes(gm)
    region = plan.region
    region_nodes = set(plan.region_nodes)
    region_nodes_tuple = plan.region_nodes_tuple
    region_live_ins = set(plan.region_live_ins)
    chunkable_live_ins = set(plan.chunkable_live_ins)

    split_live_ins: dict[fx.Node, tuple[fx.Node, fx.Node]] = {}
    dim_chunk_sizes: list[tuple[object, object]] = []
    symbol_chunk_values: dict[object, object] = {}
    symbol_full_sizes = _collect_symbol_full_sizes(gm, mode)

    # Step 6.1: planned provenance live-ins are forward live-outs from another
    # chunked region. The graph still has the original full edge, but copied
    # chunk consumers use the explicit chunk tuple built by the producer region.
    for live_in in sorted(plan.provenance_live_ins, key=order.get):
        split_live_ins[live_in] = chunk_value_nodes[live_in]

    # Step 7: record hint-derived sizes for symbolic chunk dims. Half-size
    # metadata flows through copied chunk nodes. A raw symbol always maps only
    # to that symbol's half value; full dimension expressions such as 16*u0 get
    # their own expression-level replacement so flattened batch dimensions do
    # not corrupt the meaning of u0 for later nodes.
    for live_in in chunkable_live_ins:
        val = _tensor_meta(live_in)
        assert val is not None
        hint = _annotated_chunk_hint(live_in, mode)
        if hint is None:
            continue
        dim = _chunk_dim_for_live_in(live_in, mode)
        chunk_size = _chunk_size_for_live_in(live_in, mode, val, dim)
        dim_chunk_sizes.append((val.shape[dim], chunk_size))
        for symbol in _dynamic_dim_symbols(val, dim):
            if symbol not in symbol_full_sizes:
                raise ValueError(
                    f"Chunk pass could not find a root hint for dynamic symbol "
                    f"{symbol} used by live-in {live_in.name}."
                )
            if symbol not in symbol_chunk_values:
                symbol_chunk_values[symbol] = _make_symbol_half(
                    symbol,
                    like_value=val.shape[dim],
                    symbol_full_sizes=symbol_full_sizes,
                )
    chunk_symbols = frozenset(symbol_chunk_values)

    for live_in in sorted(chunkable_live_ins, key=order.get):
        if live_in in split_live_ins:
            continue
        val = _tensor_meta(live_in)
        assert val is not None
        dim = _chunk_dim_for_live_in(live_in, mode)
        chunk_size = _chunk_size_for_live_in(
            live_in, mode, val, dim, dim_chunk_sizes=dim_chunk_sizes
        )
        # Real traced backward regions are not necessarily contiguous: untagged
        # autograd nodes can sit between module-tagged recompute nodes. Place
        # each split immediately before the first selected consumer instead of
        # assuming one contiguous region start.
        with gm.graph.inserting_before(
            _earliest_region_user(live_in, region_nodes_tuple, order)
        ):
            split_size = (
                _insert_symbolic_split_size(
                    gm,
                    live_in,
                    dim=dim,
                    meta_chunk_size=chunk_size,
                )
                if _dynamic_dim_symbols(val, dim)
                else chunk_size
            )
            split = gm.graph.call_function(
                aten.split.Tensor,
                args=(live_in, split_size, dim),
            )
            split.meta["chunk_dim"] = dim
            split.meta["chunk_size"] = chunk_size
            getitem0 = gm.graph.call_function(operator.getitem, args=(split, 0))
            getitem1 = gm.graph.call_function(operator.getitem, args=(split, 1))
            if dim != 0:
                getitem0 = gm.graph.call_function(aten.contiguous.default, (getitem0,))
                getitem1 = gm.graph.call_function(aten.contiguous.default, (getitem1,))
        getitem0.meta["val"] = _chunk_meta(val, dim, chunk_size)
        getitem1.meta["val"] = _chunk_meta(val, dim, chunk_size)
        getitem0.meta["chunk_id"] = 0
        getitem1.meta["chunk_id"] = 1
        split_live_ins[live_in] = (getitem0, getitem1)

    if not split_live_ins:
        raise ValueError(
            f"Chunk pass found no chunkable activation live-ins for region "
            f"{region.root_fqn!r}."
        )

    for live_in in sorted(region_live_ins, key=order.get):
        if live_in in split_live_ins or _tensor_meta(live_in) is not None:
            continue
        scalar_val = live_in.meta.get("val")
        if not (_free_symbols(scalar_val) & chunk_symbols):
            continue
        if not _scalar_matches_split_dim(
            scalar_val,
            mode=mode,
            split_live_ins=split_live_ins,
        ):
            raise ValueError(
                f"Chunk pass found symbolic scalar live-in {live_in.name}, but "
                "it is not a constant multiple of any split tensor dimension."
            )
        with gm.graph.inserting_before(
            _earliest_region_user(live_in, region_nodes_tuple, order)
        ):
            half_scalar = _insert_symbolic_half_scalar(
                gm,
                live_in,
                meta_half_size=scalar_val // 2,
            )
        split_live_ins[live_in] = (half_scalar, half_scalar)

    # Step 8: copy the selected closure once per chunk. Chunk 0 consumes split[0],
    # chunk 1 consumes split[1], and all copied compute is tagged with chunk_id.
    copied_by_chunk: list[dict[fx.Node, fx.Node]] = [dict(), dict()]
    for chunk_id in (0, 1):
        copied = copied_by_chunk[chunk_id]
        for node in region_nodes_tuple:
            # Copy at the original node's position. This preserves dependencies
            # on unselected interleaved nodes, such as threshold_backward.
            with gm.graph.inserting_before(node):
                new_node = gm.graph.node_copy(
                    node,
                    lambda arg: _map_arg_for_chunk(
                        arg,
                        copied=copied,
                        split_live_ins=split_live_ins,
                        chunk_id=chunk_id,
                    ),
                )
            _rename(new_node, f"{node.name}_chunk{chunk_id}")
            _copy_meta(node, new_node, chunk_id=chunk_id)
            new_node.meta["chunked_region_fqn"] = region.root_fqn
            _infer_node_val(new_node)
            chunked_val = _chunked_meta_from_original(
                node,
                dim_chunk_sizes=dim_chunk_sizes,
                symbol_chunk_values=symbol_chunk_values,
                symbol_full_sizes=symbol_full_sizes,
            )
            if chunked_val is not None:
                new_node.meta["val"] = chunked_val
            copied[node] = new_node

    replaced_live_outs = 0
    num_chunk_tuples = 0
    num_cats = 0
    num_adds = 0
    num_scalar_materializations = 0
    partial_sum_memo: dict[fx.Node, bool] = {}
    for live_out, outside_users_tuple in plan.live_out_users.items():
        val = _tensor_meta(live_out)
        outside_users = list(outside_users_tuple)
        if not outside_users:
            continue
        chunk_users = [
            user for user in outside_users if user in all_chunk_consumer_nodes
        ]
        full_users = [
            user for user in outside_users if user not in all_chunk_consumer_nodes
        ]

        if chunk_users:
            # Step 9.1: expose chunk-local live-outs explicitly for later planned
            # chunk consumers. For non-invertible values such as MoE counts, the
            # backward copy consumes these tuple elements instead of a guessed
            # split of the full materialized value.
            with gm.graph.inserting_before(_earliest_node(outside_users, order)):
                chunk_tuple = gm.graph.call_function(
                    tuple,
                    args=(
                        (copied_by_chunk[0][live_out], copied_by_chunk[1][live_out]),
                    ),
                )
                _rename(chunk_tuple, f"{live_out.name}_chunk_tuple")
                getitem0 = gm.graph.call_function(
                    operator.getitem, args=(chunk_tuple, 0)
                )
                getitem1 = gm.graph.call_function(
                    operator.getitem, args=(chunk_tuple, 1)
                )
            getitem0.meta["val"] = copied_by_chunk[0][live_out].meta.get("val")
            getitem1.meta["val"] = copied_by_chunk[1][live_out].meta.get("val")
            getitem0.meta["chunk_id"] = 0
            getitem1.meta["chunk_id"] = 1
            chunk_value_nodes[live_out] = (getitem0, getitem1)
            num_chunk_tuples += 1

        if not full_users:
            continue

        # Step 9.2: materialize only for true full consumers. Forward no-dim
        # values may be added only when explicit additive provenance proves the
        # full value is the sum of chunk values. Backward no-dim values may be
        # added only when they are graph outputs, which correspond to parameter
        # gradients in graph_trainer's traced train step.
        with gm.graph.inserting_before(_earliest_node(full_users, order)):
            if val is None:
                materialized = _materialize_scalar_live_out(
                    gm,
                    live_out,
                    copied_by_chunk[0][live_out],
                    copied_by_chunk[1][live_out],
                    mode=mode,
                    split_live_ins=split_live_ins,
                )
                num_scalar_materializations += 1
            else:
                combine_kind, dim = _combine_kind_and_dim(live_out, mode, chunk_symbols)
                if combine_kind == "add":
                    if not _can_materialize_add(
                        live_out,
                        region=region,
                        plan_nodes=plan.region_nodes,
                        chunk_symbols=chunk_symbols,
                        partial_sum_memo=partial_sum_memo,
                    ):
                        raise ValueError(
                            f"Chunk pass cannot materialize live-out "
                            f"{live_out.name} with add; expected either "
                            "forward additive provenance or a backward graph "
                            "output parameter gradient."
                        )
                    materialized = gm.graph.call_function(
                        aten.add.Tensor,
                        args=(
                            copied_by_chunk[0][live_out],
                            copied_by_chunk[1][live_out],
                        ),
                    )
                    _rename(materialized, f"{live_out.name}_chunk_sum")
                    num_adds += 1
                else:
                    materialized = gm.graph.call_function(
                        aten.cat.default,
                        args=(
                            [
                                copied_by_chunk[0][live_out],
                                copied_by_chunk[1][live_out],
                            ],
                            dim,
                        ),
                    )
                    _rename(materialized, f"{live_out.name}_chunk_cat")
                    assert dim is not None
                    materialized.meta["chunk_dim"] = dim
                    num_cats += 1
                _infer_node_val(materialized)
                _validate_materialized_meta(materialized, live_out, symbol_full_sizes)
                materialized.meta["val"] = val
                if combine_kind == "cat":
                    assert dim is not None
                    _set_materialized_chunk_dim_meta(
                        materialized,
                        live_out,
                        mode=mode,
                        dim=dim,
                        symbol_full_sizes=symbol_full_sizes,
                    )
        materialized.meta["chunked_region_fqn"] = region.root_fqn
        materialized.meta["autograd_backward"] = region.is_backward
        _set_recompute_like(live_out, materialized)
        for user in list(full_users):
            user.replace_input_with(live_out, materialized)
        replaced_live_outs += 1

    logger.debug(
        "chunk_%s transformed %s/%s: chunk_tuples=%d cats=%d adds=%d "
        "scalar_materializations=%d replaced_live_outs=%d",
        mode,
        region.root_fqn,
        "backward" if region.is_backward else "forward",
        num_chunk_tuples,
        num_cats,
        num_adds,
        num_scalar_materializations,
        replaced_live_outs,
    )
    return replaced_live_outs


def apply_chunk_pass(
    gm: fx.GraphModule,
    example_inputs: tuple[Any, ...] | None = None,
    *,
    mode: ChunkMode,
    module_patterns: list[str],
    num_static_inputs: int = 0,
    module_bucket_plans: list[list[str] | str] | None = None,
) -> fx.GraphModule:
    """Chunk selected module regions into two chunks.

    Args:
        gm: Joint fwd/loss/bwd graph.
        example_inputs: Unused, accepted for the graph pass interface.
        mode: Selects the annotated dynamic dimension to split.
        module_patterns: Dot-segment FQN patterns selecting module regions.
        num_static_inputs: Number of leading graph placeholders that represent
            model state. These are never split.
    """
    if not module_patterns:
        return gm

    if mode not in ("batch", "seq"):
        raise ValueError(f"Unknown chunk mode: {mode!r}")

    static_nodes = _static_derived_nodes(
        gm, _static_placeholders(gm, num_static_inputs)
    )
    _propagate_chunk_dim_metadata(gm)
    regions = _find_regions(gm, module_patterns)
    if not regions:
        raise ValueError(
            f"No graph regions matched chunk_{mode} patterns: {module_patterns}"
        )
    _validate_regions_within_boundaries(
        regions,
        module_bucket_plans=module_bucket_plans,
    )
    logger.debug(
        "chunk_%s matched regions: %s",
        mode,
        [
            (region.root_fqn, "backward" if region.is_backward else "forward")
            for region in regions
        ],
    )
    plans = _plan_regions(
        gm,
        regions,
        mode=mode,
        static_nodes=static_nodes,
    )
    all_chunk_consumer_nodes = _chunk_consumer_nodes(plans)
    chunk_value_nodes: dict[fx.Node, tuple[fx.Node, fx.Node]] = {}

    transformed = 0
    for plan in plans:
        logger.debug(
            "chunk_%s plan %s/%s: search_space=%d chunkable_live_ins=%d "
            "provenance_live_ins=%d copied_nodes=%d live_outs=%d",
            mode,
            plan.region.root_fqn,
            "backward" if plan.region.is_backward else "forward",
            len(plan.search_space),
            len(plan.chunkable_live_ins),
            len(plan.provenance_live_ins),
            len(plan.region_nodes),
            len(plan.live_out_users),
        )
        transformed += _transform_region(
            gm,
            plan,
            mode=mode,
            all_chunk_consumer_nodes=all_chunk_consumer_nodes,
            chunk_value_nodes=chunk_value_nodes,
        )

    order = _ordered_nodes(gm)
    planned_nodes = tuple(sorted(all_chunk_consumer_nodes, key=order.__getitem__))
    _safe_erase_region(planned_nodes)

    gm.graph.lint()
    gm.recompile()
    logger.info(
        "Applied chunk_%s to %d regions (%d materialized live-outs): %s",
        mode,
        len(regions),
        transformed,
        module_patterns,
    )
    return gm


def chunk_batch_pass(
    gm: fx.GraphModule,
    example_inputs: tuple[Any, ...] | None = None,
    *,
    module_patterns: list[str],
    num_static_inputs: int = 0,
    module_bucket_plans: list[list[str] | str] | None = None,
) -> fx.GraphModule:
    return apply_chunk_pass(
        gm,
        example_inputs,
        mode="batch",
        module_patterns=module_patterns,
        num_static_inputs=num_static_inputs,
        module_bucket_plans=module_bucket_plans,
    )


def chunk_seq_pass(
    gm: fx.GraphModule,
    example_inputs: tuple[Any, ...] | None = None,
    *,
    module_patterns: list[str],
    num_static_inputs: int = 0,
    module_bucket_plans: list[list[str] | str] | None = None,
) -> fx.GraphModule:
    return apply_chunk_pass(
        gm,
        example_inputs,
        mode="seq",
        module_patterns=module_patterns,
        num_static_inputs=num_static_inputs,
        module_bucket_plans=module_bucket_plans,
    )


def _custom_meta(node: fx.Node) -> dict[str, Any]:
    custom = node.meta.get("custom")
    return custom if isinstance(custom, dict) else {}


def _ep_region(node: fx.Node) -> str | None:
    ep = _custom_meta(node).get("EP")
    return ep if ep in ("dispatch", "combine") else None


def _is_all_to_all_node(node: fx.Node) -> bool:
    return node.op == "call_function" and "all_to_all_single" in str(node.target)


def _find_last_all_to_all(nodes: list[fx.Node], order: dict[fx.Node, int]) -> fx.Node:
    all_to_all_nodes = [node for node in nodes if _is_all_to_all_node(node)]
    if not all_to_all_nodes:
        raise ValueError(
            "ep_overlap expected an all_to_all_single node in EP region nodes: "
            f"{[node.name for node in nodes]}"
        )
    return max(all_to_all_nodes, key=order.__getitem__)


def _phase_nodes(
    chunk_nodes: list[fx.Node],
    *,
    is_backward: bool,
    order: dict[fx.Node, int],
) -> tuple[list[fx.Node], list[fx.Node], list[fx.Node]] | None:
    dispatch_nodes = [node for node in chunk_nodes if _ep_region(node) == "dispatch"]
    combine_nodes = [node for node in chunk_nodes if _ep_region(node) == "combine"]
    if not dispatch_nodes and not combine_nodes:
        return None
    if not dispatch_nodes or not combine_nodes:
        raise ValueError(
            "ep_overlap requires both EP dispatch and combine regions in each "
            f"chunk, found dispatch={bool(dispatch_nodes)} combine={bool(combine_nodes)}"
        )

    last_dispatch = _find_last_all_to_all(dispatch_nodes, order)
    last_combine = _find_last_all_to_all(combine_nodes, order)
    first_launch, second_launch = (
        (last_combine, last_dispatch) if is_backward else (last_dispatch, last_combine)
    )
    first_idx = order[first_launch]
    second_idx = order[second_launch]
    if first_idx >= second_idx:
        expected = "combine-to-dispatch" if is_backward else "dispatch-to-combine"
        raise ValueError(
            f"ep_overlap expected {expected} all-to-all order, got "
            f"{first_launch.name} after {second_launch.name}"
        )

    for ep_nodes, last_launch in (
        (dispatch_nodes, last_dispatch),
        (combine_nodes, last_combine),
    ):
        last_idx = order[last_launch]
        for node in ep_nodes:
            if order[node] > last_idx:
                custom = dict(_custom_meta(node))
                custom["EP_wait"] = True
                node.meta["custom"] = custom

    return (
        [node for node in chunk_nodes if order[node] <= first_idx],
        [node for node in chunk_nodes if first_idx < order[node] <= second_idx],
        [node for node in chunk_nodes if order[node] > second_idx],
    )


def _add_phase_deps(
    deps: dict[fx.Node, OrderedSet[fx.Node]],
    before: list[fx.Node],
    after: list[fx.Node],
) -> None:
    for node in after:
        node_deps = deps.setdefault(node, OrderedSet())
        for dep in before:
            if dep is not node:
                node_deps.add(dep)


def _schedule_ep_overlap_regions(
    gm: fx.GraphModule,
    *,
    module_patterns: list[str],
    require_all_to_all: bool,
) -> int:
    """Order chunked regions so EP all-to-all launches precede peer waits.

    Each chunk is split into three scheduling phases:
    1. compute through the first EP all-to-all launch,
    2. first wait suffix, expert compute, and second EP all-to-all launch,
    3. second wait suffix and post-EP tail.

    Forward regions use dispatch then combine. Backward autograd sees combine
    gradients before dispatch gradients, so it uses combine then dispatch and
    reverses the chunk order. Non-MoE transformer blocks have no EP phases and
    are left in the chunk pass's original topological order.
    """
    order = _ordered_nodes(gm)
    grouped: dict[tuple[str, bool], dict[int, list[fx.Node]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for node in gm.graph.nodes:
        chunk_id = node.meta.get("chunk_id")
        if chunk_id not in (0, 1):
            continue
        root = node.meta.get("chunked_region_fqn")
        if not isinstance(root, str) or not root:
            fqn = _fqn(node)
            roots = [match for p in module_patterns if (match := _pattern_root(p, fqn))]
            root = roots[0] if len(roots) == 1 else ""
        if not root:
            continue
        grouped[(root, _is_backward_node(node))][chunk_id].append(node)

    deps: dict[fx.Node, OrderedSet[fx.Node]] = {}
    scheduled = 0
    for (root, is_backward), by_chunk in sorted(
        grouped.items(),
        key=lambda item: min(order[n] for nodes in item[1].values() for n in nodes),
    ):
        if set(by_chunk) != {0, 1}:
            raise ValueError(
                f"ep_overlap expected both chunk 0 and chunk 1 for region {root!r} "
                f"({'backward' if is_backward else 'forward'}), found {sorted(by_chunk)}"
            )

        chunk_order = (1, 0) if is_backward else (0, 1)
        phases_by_chunk = {
            chunk_id: _phase_nodes(
                sorted(by_chunk[chunk_id], key=order.__getitem__),
                is_backward=is_backward,
                order=order,
            )
            for chunk_id in chunk_order
        }
        missing_phases = [
            chunk_id for chunk_id, phases in phases_by_chunk.items() if phases is None
        ]
        if missing_phases:
            if len(missing_phases) == len(phases_by_chunk):
                continue
            raise ValueError(
                f"ep_overlap found EP all-to-all regions for only one chunk of "
                f"{root!r} ({'backward' if is_backward else 'forward'}): "
                f"missing chunks {missing_phases}."
            )

        assert phases_by_chunk[chunk_order[0]] is not None
        assert phases_by_chunk[chunk_order[1]] is not None
        first = phases_by_chunk[chunk_order[0]]
        second = phases_by_chunk[chunk_order[1]]
        assert first is not None and second is not None
        ordered_phases = [
            first[0],
            second[0],
            first[1],
            second[1],
            first[2],
            second[2],
        ]
        non_empty_phases = [phase for phase in ordered_phases if phase]
        for earlier, later in zip(non_empty_phases, non_empty_phases[1:]):
            _add_phase_deps(deps, earlier, later)
        scheduled += 1

    if scheduled:
        _stable_topological_sort(gm.graph, deps)
        gm.graph.lint()
        gm.recompile()
    elif require_all_to_all:
        raise ValueError(
            f"ep_overlap did not find any chunked EP all-to-all regions for "
            f"patterns {module_patterns}."
        )
    return scheduled


def ep_overlap_pass(
    gm: fx.GraphModule,
    example_inputs: tuple[Any, ...] | None = None,
    *,
    mode: ChunkMode,
    module_patterns: list[str],
    num_static_inputs: int = 0,
    module_bucket_plans: list[list[str] | str] | None = None,
    require_all_to_all: bool = True,
) -> fx.GraphModule:
    """Chunk selected regions and reorder chunk streams around EP all-to-alls."""
    gm = apply_chunk_pass(
        gm,
        example_inputs,
        mode=mode,
        module_patterns=module_patterns,
        num_static_inputs=num_static_inputs,
        module_bucket_plans=module_bucket_plans,
    )
    scheduled = _schedule_ep_overlap_regions(
        gm,
        module_patterns=module_patterns,
        require_all_to_all=require_all_to_all,
    )
    logger.info(
        "Applied ep_overlap to %d chunked region(s): mode=%s modules=%s",
        scheduled,
        mode,
        module_patterns,
    )
    return gm
