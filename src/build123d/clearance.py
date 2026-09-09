"""
build123d clearance

name: clearance.py
by:   Netra
date: September 9th 2026

license:

    Copyright 2026 The build123d Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

"""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Any

from build123d.geometry import TOLERANCE, Axis, Vector
from build123d.probe import ProbeResult, ProbedBody, probe
from build123d.topology.shape_core import Shape

__all__ = [
    "GapRecord",
    "SweepStep",
    "SweepPath",
    "ClearanceResult",
    "clearance",
]


@dataclass(frozen=True)
class GapRecord:
    """Unordered pair of bodies and the real minimum distance between them."""

    a: str
    b: str
    gap: float
    point_a: Vector
    point_b: Vector
    ok: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready pair record."""
        return {
            "a": self.a,
            "b": self.b,
            "gap": self.gap,
            "point_a": [self.point_a.X, self.point_a.Y, self.point_a.Z],
            "point_b": [self.point_b.X, self.point_b.Y, self.point_b.Z],
            "ok": self.ok,
        }


@dataclass(frozen=True)
class SweepStep:
    """One offset along the insertion axis."""

    offset: float
    colliding: tuple[str, ...]
    ok: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready step record."""
        return {
            "offset": self.offset,
            "colliding": list(self.colliding),
            "ok": self.ok,
        }


@dataclass(frozen=True)
class SweepPath:
    """Discrete insertion of one body along an axis."""

    moving: str
    axis: Vector
    travel: float
    steps: tuple[SweepStep, ...]

    @property
    def ok(self) -> bool:
        """True when no step has a volume intersect."""
        return all(step.ok for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready sweep record."""
        return {
            "moving": self.moving,
            "axis": [self.axis.X, self.axis.Y, self.axis.Z],
            "travel": self.travel,
            "steps": [step.to_dict() for step in self.steps],
            "ok": self.ok,
        }


@dataclass(frozen=True)
class ClearanceResult:
    """Min-gap matrix plus an optional insertion sweep."""

    slip: float
    pairs: tuple[GapRecord, ...]
    sweep: SweepPath | None

    @property
    def ok(self) -> bool:
        """True when every pair meets ``slip`` and the sweep (if any) is clear."""
        return all(pair.ok for pair in self.pairs) and (
            self.sweep is None or self.sweep.ok
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready clearance report."""
        return {
            "slip": self.slip,
            "ok": self.ok,
            "pairs": [pair.to_dict() for pair in self.pairs],
            "sweep": None if self.sweep is None else self.sweep.to_dict(),
        }


def clearance(
    source: PathLike | str | bytes | Shape | ProbeResult,
    *,
    slip: float,
    moving: str | int | None = None,
    axis: Axis | None = None,
    travel: float | None = None,
    steps: int = 8,
) -> ClearanceResult:
    """Measure pairwise gaps and optionally sweep one body along an axis.

    Args:
        source: STEP path, Shape, or a :class:`ProbeResult` from
            :func:`probe`. Bodies follow probe inventory order.
        slip: Minimum allowed gap. A pair passes when
            ``gap + TOLERANCE >= slip``.
        moving: Body to translate during the sweep. A string matches
            ``ProbedBody.name``. An int is a 0-based inventory index.
        axis: Insertion axis. The body is translated along
            ``axis.direction``.
        travel: Inclusive path length from the given pose to the last
            step. Must be positive when a sweep is requested.
        steps: Inclusive sample count along ``[0, travel]``. Must be at
            least 2 so a mid-path hit cannot hide behind the endpoints.

    Returns:
        ClearanceResult: Pairwise gaps always. ``sweep`` is set only
        when ``moving``, ``axis``, and ``travel`` are all given. The
        library does not raise when a check fails; read ``ok``.
    """
    slip_value = _require_slip(slip)
    inventory = source if isinstance(source, ProbeResult) else probe(source)
    labeled = _labeled_bodies(inventory.bodies)
    pairs = _gap_matrix(labeled, slip_value)
    sweep = _sweep_path(labeled, moving, axis, travel, steps)
    return ClearanceResult(slip=slip_value, pairs=pairs, sweep=sweep)


def _require_slip(slip: float) -> float:
    if not isinstance(slip, (int, float)) or isinstance(slip, bool):
        raise ValueError("slip must be a number >= 0")
    if slip < 0:
        raise ValueError("slip must be a number >= 0")
    return float(slip)


def _labeled_bodies(
    bodies: tuple[ProbedBody, ...],
) -> tuple[tuple[str, Shape], ...]:
    labeled: list[tuple[str, Shape]] = []
    for index, body in enumerate(bodies):
        name = body.name if body.name else f"body[{index}]"
        labeled.append((name, body.shape))
    return tuple(labeled)


def _gap_matrix(
    bodies: tuple[tuple[str, Shape], ...], slip: float
) -> tuple[GapRecord, ...]:
    records: list[GapRecord] = []
    for i, (name_a, shape_a) in enumerate(bodies):
        for name_b, shape_b in bodies[i + 1 :]:
            gap, point_a, point_b = shape_a.distance_to_with_closest_points(shape_b)
            records.append(
                GapRecord(
                    a=name_a,
                    b=name_b,
                    gap=gap,
                    point_a=point_a,
                    point_b=point_b,
                    ok=gap + TOLERANCE >= slip,
                )
            )
    records.sort(key=lambda record: (record.a, record.b))
    return tuple(records)


def _sweep_requested(
    moving: str | int | None, axis: Axis | None, travel: float | None
) -> bool:
    return moving is not None or axis is not None or travel is not None


def _sweep_path(
    bodies: tuple[tuple[str, Shape], ...],
    moving: str | int | None,
    axis: Axis | None,
    travel: float | None,
    steps: int,
) -> SweepPath | None:
    if not _sweep_requested(moving, axis, travel):
        return None
    if moving is None or axis is None or travel is None:
        raise ValueError("sweep requires moving, axis, and travel")
    if not isinstance(travel, (int, float)) or isinstance(travel, bool) or travel <= 0:
        raise ValueError("travel must be a number > 0")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 2:
        raise ValueError("steps must be an int >= 2")

    moving_name, moving_shape = _resolve_moving(bodies, moving)
    others = [(name, shape) for name, shape in bodies if name != moving_name]
    direction = Vector(axis.direction).normalized()
    span = float(travel)
    samples = [i * span / (steps - 1) for i in range(steps)]
    sweep_steps = tuple(
        _sweep_step(offset, moving_shape, direction, others) for offset in samples
    )
    return SweepPath(
        moving=moving_name,
        axis=direction,
        travel=span,
        steps=sweep_steps,
    )


def _resolve_moving(
    bodies: tuple[tuple[str, Shape], ...], moving: str | int
) -> tuple[str, Shape]:
    if isinstance(moving, bool) or not isinstance(moving, (str, int)):
        raise ValueError("moving must be a body name or inventory index")
    if isinstance(moving, int):
        if moving < 0 or moving >= len(bodies):
            raise ValueError(f"moving index {moving} is out of range")
        return bodies[moving]
    for name, shape in bodies:
        if name == moving:
            return name, shape
    raise ValueError(f"moving body not found: {moving}")


def _sweep_step(
    offset: float,
    moving_shape: Shape,
    direction: Vector,
    others: list[tuple[str, Shape]],
) -> SweepStep:
    placed = moving_shape if offset == 0 else moving_shape.translate(direction * offset)
    colliding = tuple(
        name for name, shape in others if placed.intersect(shape) is not None
    )
    return SweepStep(offset=offset, colliding=colliding, ok=not colliding)
