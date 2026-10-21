"""Variable Elimination desde cero sobre la red bayesiana.

Operaciones implementadas:
- reducción de factor por evidencia (slicing en el eje correspondiente)
- producto de factores con broadcasting por alineación de variables
- marginalización (suma sobre eje de la variable eliminada)
- normalización final del factor resultante

Heurística de orden de eliminación: min-degree (variable con menor número
de factores en los que participa). Suficiente para redes pequeñas.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

import numpy as np

from .network import BayesianNetwork, Factor


def reduce_by_evidence(factor: Factor, evidence: Mapping[str, int]) -> Factor:
    array = factor.array
    vars_remaining = list(factor.variables)
    for axis in range(len(factor.variables) - 1, -1, -1):
        var = factor.variables[axis]
        if var in evidence:
            array = np.take(array, indices=evidence[var], axis=axis)
            vars_remaining.remove(var)
    if array.ndim == 0:
        array = array.reshape(())
    return Factor(variables=vars_remaining, array=np.asarray(array))


def multiply(f1: Factor, f2: Factor) -> Factor:
    union: List[str] = list(f1.variables)
    for v in f2.variables:
        if v not in union:
            union.append(v)

    def aligned(factor: Factor) -> np.ndarray:
        if not union:
            return factor.array
        if factor.array.ndim == 0:
            return factor.array.reshape([1] * len(union))
        perm = sorted(
            range(factor.array.ndim),
            key=lambda i: union.index(factor.variables[i]),
        )
        arr_t = factor.array.transpose(perm)
        var_to_size = {
            v: factor.array.shape[i] for i, v in enumerate(factor.variables)
        }
        shape = [var_to_size.get(v, 1) for v in union]
        return arr_t.reshape(shape)

    a1 = aligned(f1)
    a2 = aligned(f2)
    return Factor(variables=union, array=a1 * a2)


def marginalize(factor: Factor, var: str) -> Factor:
    axis = factor.variables.index(var)
    new_vars = [v for v in factor.variables if v != var]
    summed = factor.array.sum(axis=axis)
    return Factor(variables=new_vars, array=summed)


def _pick_next(vars_to_eliminate: Sequence[str], factors: Sequence[Factor]) -> str:
    return min(
        vars_to_eliminate,
        key=lambda v: sum(1 for f in factors if v in f.variables),
    )


def variable_elimination(
    net: BayesianNetwork,
    query: Sequence[str],
    evidence: Mapping[str, int] | None = None,
) -> Factor:
    """Devuelve P(query | evidence) como Factor normalizado."""
    evidence = dict(evidence or {})
    factors: List[Factor] = [reduce_by_evidence(f, evidence) for f in net.factors()]
    hidden = [n for n in net.nodes if n not in query and n not in evidence]

    while hidden:
        var = _pick_next(hidden, factors)
        hidden.remove(var)
        involved = [f for f in factors if var in f.variables]
        rest = [f for f in factors if var not in f.variables]
        if not involved:
            continue
        product = involved[0]
        for f in involved[1:]:
            product = multiply(product, f)
        summed = marginalize(product, var)
        factors = rest + [summed]

    result = factors[0]
    for f in factors[1:]:
        result = multiply(result, f)
    total = result.array.sum()
    if total > 0:
        result = Factor(variables=result.variables, array=result.array / total)
    return result
