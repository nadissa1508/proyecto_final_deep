"""Backtracking Search con Forward Checking y heurística MRV.

Implementado desde cero (sin librerías de CSP). Resuelve la asignación de
variedades de café a un conjunto de parcelas. Restricciones duras: altitud,
pH, sombra. Restricción inter-parcelas opcional: AllDifferent. Optimiza una
suma ponderada de restricciones blandas (resistencia a roya, rendimiento)
explorando todas las soluciones consistentes y devolviendo la de mayor score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .varieties_loader import Variety, satisfies_hard_constraints


@dataclass(frozen=True)
class Parcela:
    parcela_id: str
    altitud: float
    ph: float
    sombra_disponible: int


@dataclass
class Assignment:
    parcela_id: str
    variety: str
    score: float
    hard_passed: list[str]


@dataclass
class Solution:
    assignments: list[Assignment]
    total_score: float
    nodes_expanded: int = 0


@dataclass
class _SearchState:
    domains: dict[str, list[Variety]]
    assignment: dict[str, Variety] = field(default_factory=dict)
    best: Solution | None = None
    nodes: int = 0


def _soft_score(variety: Variety, w_rust: float, w_yield: float) -> float:
    return (w_rust * variety.rust_resistance + w_yield * variety.yield_score) / 10.0


def _initial_domains(
    parcelas: list[Parcela], varieties: list[Variety],
) -> dict[str, list[Variety]]:
    return {
        p.parcela_id: [
            v for v in varieties
            if satisfies_hard_constraints(v, p.altitud, p.ph, p.sombra_disponible)
        ]
        for p in parcelas
    }


def _select_unassigned_mrv(
    parcelas: list[Parcela], state: _SearchState,
) -> Parcela | None:
    unassigned = [p for p in parcelas if p.parcela_id not in state.assignment]
    if not unassigned:
        return None
    return min(unassigned, key=lambda p: len(state.domains[p.parcela_id]))


def _forward_check(
    target_id: str, chosen: Variety, state: _SearchState, diversify: bool,
) -> dict[str, list[Variety]] | None:
    """Devuelve nuevos dominios filtrados (o None si alguno queda vacío)."""
    if not diversify:
        return {pid: list(dom) for pid, dom in state.domains.items()}
    new_domains: dict[str, list[Variety]] = {}
    for pid, dom in state.domains.items():
        if pid == target_id or pid in state.assignment:
            new_domains[pid] = list(dom)
            continue
        pruned = [v for v in dom if v.name != chosen.name]
        if not pruned:
            return None
        new_domains[pid] = pruned
    return new_domains


def _backtrack(
    parcelas: list[Parcela], state: _SearchState,
    w_rust: float, w_yield: float, diversify: bool,
) -> None:
    state.nodes += 1
    parcela = _select_unassigned_mrv(parcelas, state)
    if parcela is None:
        total = sum(
            _soft_score(v, w_rust, w_yield) for v in state.assignment.values()
        )
        if state.best is None or total > state.best.total_score:
            state.best = Solution(
                assignments=[
                    Assignment(
                        parcela_id=p.parcela_id,
                        variety=state.assignment[p.parcela_id].name,
                        score=_soft_score(
                            state.assignment[p.parcela_id], w_rust, w_yield,
                        ),
                        hard_passed=["altitud", "ph", "sombra"],
                    )
                    for p in parcelas
                ],
                total_score=total,
            )
        return

    candidates = sorted(
        state.domains[parcela.parcela_id],
        key=lambda v: _soft_score(v, w_rust, w_yield),
        reverse=True,
    )
    for variety in candidates:
        new_domains = _forward_check(
            parcela.parcela_id, variety, state, diversify,
        )
        if new_domains is None:
            continue
        saved_domains = state.domains
        state.domains = new_domains
        state.assignment[parcela.parcela_id] = variety
        _backtrack(parcelas, state, w_rust, w_yield, diversify)
        del state.assignment[parcela.parcela_id]
        state.domains = saved_domains


def solve(
    parcelas: Iterable[Parcela],
    varieties: list[Variety],
    *,
    diversify: bool = True,
    w_rust: float = 0.5,
    w_yield: float = 0.5,
) -> Solution | None:
    parcelas = list(parcelas)
    if not parcelas:
        return Solution(assignments=[], total_score=0.0)
    state = _SearchState(domains=_initial_domains(parcelas, varieties))
    if any(not dom for dom in state.domains.values()):
        empty = [pid for pid, dom in state.domains.items() if not dom]
        raise ValueError(f"Parcelas sin variedad compatible: {empty}")
    _backtrack(parcelas, state, w_rust, w_yield, diversify)
    if state.best is not None:
        state.best.nodes_expanded = state.nodes
    return state.best
