"""Compute cost and FLOPs exporting utilities for JAX models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Union
import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class FLOPsReport:
    """Detailed compute cost and FLOPs report for a model."""

    total_flops: int
    total_macs: int
    param_count: int
    op_breakdown: Dict[str, int] = field(default_factory=dict)

    @property
    def gflops(self) -> float:
        """Floating point operations in GigaFLOPs (10^9)."""
        return self.total_flops / 1e9

    @property
    def mflops(self) -> float:
        """Floating point operations in MegaFLOPs (10^6)."""
        return self.total_flops / 1e6

    @property
    def gmacs(self) -> float:
        """Multiply-accumulate operations in GMACs (10^9)."""
        return self.total_macs / 1e9

    @property
    def param_count_m(self) -> float:
        """Parameter count in millions."""
        return self.param_count / 1e6

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "total_flops": self.total_flops,
            "gflops": round(self.gflops, 4),
            "mflops": round(self.mflops, 2),
            "total_macs": self.total_macs,
            "gmacs": round(self.gmacs, 4),
            "param_count": self.param_count,
            "param_count_m": round(self.param_count_m, 2),
            "op_breakdown": self.op_breakdown,
        }

    def summary(self) -> str:
        """Format a human-readable table summary."""
        lines = [
            "================ FLOPs & Compute Cost Report ================",
            f" Total FLOPs:        {self.total_flops:,} ({self.gflops:.3f} GFLOPs / {self.mflops:.1f} MFLOPs)",
            f" Total MACs:         {self.total_macs:,} ({self.gmacs:.3f} GMACs)",
            f" Total Parameters:   {self.param_count:,} ({self.param_count_m:.2f} M)",
            "----------------------- Op Breakdown ------------------------",
        ]
        for op, count in sorted(self.op_breakdown.items(), key=lambda x: x[1], reverse=True):
            pct = (count / self.total_flops * 100.0) if self.total_flops > 0 else 0.0
            lines.append(f" {op:<20} {count:>16,} FLOPs ({pct:>5.1f}%)")
        lines.append("=============================================================")
        return "\n".join(lines)


def _calc_dot_general_flops(eqn: Any) -> int:
    """Calculate FLOPs for jax.lax.dot_general primitive."""
    lhs_shape = eqn.invars[0].aval.shape
    rhs_shape = eqn.invars[1].aval.shape
    out_shape = eqn.outvars[0].aval.shape

    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = eqn.params["dimension_numbers"]

    contract_size = 1
    for d in lhs_contract:
        contract_size *= lhs_shape[d]

    out_size = 1
    for s in out_shape:
        out_size *= s

    # 1 MAC = 1 multiply + 1 accumulate = 2 FLOPs
    return 2 * contract_size * out_size


def _calc_conv_general_dilated_flops(eqn: Any) -> int:
    """Calculate FLOPs for jax.lax.conv_general_dilated primitive."""
    rhs_shape = eqn.invars[1].aval.shape
    out_shape = eqn.outvars[0].aval.shape
    dnum = eqn.params["dimension_numbers"]

    # Number of output elements
    out_size = 1
    for s in out_shape:
        out_size *= s

    # Kernel spatial size
    kernel_spatial = 1
    for d in dnum.rhs_spec[2:]:
        kernel_spatial *= rhs_shape[d]

    # In channels per group
    in_channels = rhs_shape[dnum.rhs_spec[1]]

    return 2 * kernel_spatial * in_channels * out_size


def count_parameters(params: Any) -> int:
    """Count total number of scalar parameters in a PyTree."""
    leaves = jax.tree_util.tree_leaves(params)
    return sum(int(np.prod(x.shape)) for x in leaves if hasattr(x, "shape"))


def export_flops(
    apply_fn: Callable[..., Any],
    *sample_args: Any,
    params: Optional[Any] = None,
    verbose: bool = False,
) -> FLOPsReport:
    """
    Export exact FLOPs and compute cost for any JAX function or model apply.

    Args:
        apply_fn: Function to evaluate (e.g. `lambda x: model.apply(params, x)`)
        *sample_args: Sample input arrays with expected batch size and shape.
        params: Optional parameter PyTree to count parameters.
        verbose: If True, print summary to console.

    Returns:
        FLOPsReport with FLOPs, MACs, parameters, and operation breakdown.
    """
    closed_jaxpr = jax.make_jaxpr(apply_fn)(*sample_args)

    op_breakdown: Dict[str, int] = {}
    total_flops = 0

    for eqn in closed_jaxpr.eqns:
        p_name = eqn.primitive.name
        f = 0

        if p_name == "dot_general":
            f = _calc_dot_general_flops(eqn)
        elif p_name == "conv_general_dilated":
            f = _calc_conv_general_dilated_flops(eqn)
        elif p_name in ("add", "sub", "mul", "div", "neg"):
            out_size = 1
            for s in eqn.outvars[0].aval.shape:
                out_size *= s
            f = out_size
        elif p_name in ("exp", "log", "sin", "cos", "tanh"):
            out_size = 1
            for s in eqn.outvars[0].aval.shape:
                out_size *= s
            f = 4 * out_size  # Transcendental ops typically 4-8 FLOPs

        if f > 0:
            op_breakdown[p_name] = op_breakdown.get(p_name, 0) + f
            total_flops += f

    # MACs are predominantly from matrix multiplies and convolutions (2 FLOPs per MAC)
    dot_and_conv_flops = op_breakdown.get("dot_general", 0) + op_breakdown.get("conv_general_dilated", 0)
    total_macs = dot_and_conv_flops // 2

    # Parameter count
    param_cnt = count_parameters(params) if params is not None else 0
    if param_cnt == 0:
        # Check if first sample_arg looks like params dict
        if len(sample_args) > 0 and isinstance(sample_args[0], dict):
            param_cnt = count_parameters(sample_args[0])

    report = FLOPsReport(
        total_flops=total_flops,
        total_macs=total_macs,
        param_count=param_cnt,
        op_breakdown=op_breakdown,
    )

    if verbose:
        print(report.summary())

    return report
