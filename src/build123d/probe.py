"""
build123d probe

name: probe.py
by:   Netra
date: September 9th 2026

desc:
    Library-first STEP inventory. Trust measured geometry: body names, bounding
    boxes, and cylindrical holes (center, axis, diameter). Strip contaminating
    bodies by exact label. STEP metadata is treated as untrusted.

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

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from math import pi
from os import PathLike
from typing import Any

from build123d.build_enums import GeomType
from build123d.geometry import TOLERANCE, Axis, BoundBox, Vector
from build123d.importers import import_step
from build123d.topology.shape_core import Shape
from build123d.topology.two_d import Face

__all__ = [
    "ProbedHole",
    "ProbedBody",
    "ProbeResult",
    "probe",
]

# Split faces of one hole share an axis, a radius, and an axial extent.
# Grouped span below π is a cylindrical fillet, not a hole.
_AXIS_ANGLE_TOL_DEG = 0.1
_AXIS_LINEAR_TOL = 1e-4
_AXIS_GAP_TOL = 1e-3
_MIN_HOLE_SPAN_RAD = pi


@dataclass(frozen=True)
class ProbedHole:
    """A cylindrical hole recovered from faces, not from STEP metadata.

    ``axis`` is a unit direction. It is canonicalized so the dominant
    component is non-negative (preferring +Z, then +Y, then +X) so values
    are stable to hard-code after a visual check.
    """

    center: Vector
    axis: Vector
    diameter: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready hole record."""
        return {
            "center": [self.center.X, self.center.Y, self.center.Z],
            "axis": [self.axis.X, self.axis.Y, self.axis.Z],
            "diameter": self.diameter,
        }


@dataclass(frozen=True)
class ProbedBody:
    """One solid/body after optional name strip."""

    name: str
    bbox: BoundBox
    holes: tuple[ProbedHole, ...]
    shape: Shape

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready body record (geometry only; ``shape`` is omitted)."""
        return {
            "name": self.name,
            "bbox": {
                "min": [self.bbox.min.X, self.bbox.min.Y, self.bbox.min.Z],
                "max": [self.bbox.max.X, self.bbox.max.Y, self.bbox.max.Z],
            },
            "holes": [hole.to_dict() for hole in self.holes],
        }


@dataclass(frozen=True)
class ProbeResult:
    """Inventory of remaining bodies, in assembly-tree order."""

    bodies: tuple[ProbedBody, ...]

    @property
    def names(self) -> tuple[str, ...]:
        """Body labels in report order (empty string if unlabeled)."""
        return tuple(body.name for body in self.bodies)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready inventory."""
        return {"bodies": [body.to_dict() for body in self.bodies]}


def probe(
    source: PathLike | str | bytes | Shape,
    strip: str | Iterable[str] | None = None,
) -> ProbeResult:
    """Inventory bodies and cylindrical holes from a STEP or Shape.

    Use this before designing around an imported part. STEP files often
    include extra bodies and names that do not match the geometry; probe
    measures the solids instead.

    Args:
        source: STEP path or an already-loaded :class:`Shape` (typically
            the result of :func:`import_step`).
        strip: Body names to drop. A single string or an iterable of
            strings. Match rule is **exact** equality against the body's
            ``Shape.label`` after import (case-sensitive). ``import_step``
            sanitizes labels by replacing space, ``.``, ``(``, and ``)``
            with ``_`` — probe first, then strip using the names it
            reports. Names that match no body are ignored.

    Returns:
        ProbeResult: Remaining bodies with names, axis-aligned bounding
        boxes, and holes. Each hole has ``center``, unit ``axis``, and
        ``diameter``. An empty ``bodies`` tuple means nothing remained
        (including after strip); the library does not raise in that case.
    """

    root = source if isinstance(source, Shape) else import_step(source)
    excluded = _strip_names(strip)
    bodies: list[ProbedBody] = []
    for name, body in _iter_bodies(root):
        if name in excluded:
            continue
        bodies.append(
            ProbedBody(
                name=name,
                bbox=body.bounding_box(),
                holes=_detect_holes(body),
                shape=body,
            )
        )
    return ProbeResult(bodies=tuple(bodies))


def _strip_names(strip: str | Iterable[str] | None) -> set[str]:
    if strip is None:
        return set()
    if isinstance(strip, str):
        return {strip}
    return set(strip)


def _iter_bodies(shape: Shape) -> Iterator[tuple[str, Shape]]:
    """Yield ``(name, shape)`` for each assembly leaf or topological solid."""
    children = list(getattr(shape, "children", ()) or ())
    if children:
        for child in children:
            yield from _iter_bodies(child)
        return

    solids = list(shape.solids())
    if not solids:
        return
    if len(solids) == 1:
        solid = solids[0]
        yield shape.label or solid.label, shape
        return
    for solid in solids:
        yield solid.label, solid


def _detect_holes(shape: Shape) -> tuple[ProbedHole, ...]:
    """Cylindrical faces that curve into the solid, grouped into holes."""
    groups: list[tuple[Axis, float, list[Face]]] = []
    for face in shape.faces():
        if face.geom_type != GeomType.CYLINDER:
            continue
        try:
            if not face.is_circular_concave:
                continue
        except ValueError:
            continue
        axis = face.axis_of_rotation
        if axis is None:
            continue
        radius = _cylinder_radius(face, axis)
        if radius is None or radius <= TOLERANCE:
            continue
        for group_axis, group_radius, group_faces in groups:
            if not _same_hole(group_axis, group_radius, axis, radius):
                continue
            if not _axially_adjacent(group_faces, [face], group_axis):
                continue
            group_faces.append(face)
            break
        else:
            groups.append((axis, radius, [face]))

    holes = [
        _hole_from_group(axis, radius, faces)
        for axis, radius, faces in groups
        if _angular_span(faces) + 1e-6 >= _MIN_HOLE_SPAN_RAD
    ]
    holes.sort(
        key=lambda hole: (hole.diameter, hole.center.X, hole.center.Y, hole.center.Z)
    )
    return tuple(holes)


def _same_hole(
    first_axis: Axis, first_radius: float, second_axis: Axis, second_radius: float
) -> bool:
    radius_tol = max(TOLERANCE, 1e-6 * max(first_radius, second_radius))
    if abs(first_radius - second_radius) > radius_tol:
        return False
    return first_axis.is_parallel(
        second_axis, angular_tolerance=_AXIS_ANGLE_TOL_DEG
    ) and not first_axis.is_skew(second_axis, tolerance=_AXIS_LINEAR_TOL)


def _axial_params(faces: Iterable[Face], axis: Axis) -> list[float]:
    direction = axis.direction
    params: list[float] = []
    for face in faces:
        vertices = list(face.vertices())
        if vertices:
            params.extend(
                (Vector(vertex.X, vertex.Y, vertex.Z) - axis.position).dot(direction)
                for vertex in vertices
            )
        else:
            params.append((face.center() - axis.position).dot(direction))
    return params


def _axial_range(faces: Iterable[Face], axis: Axis) -> tuple[float, float] | None:
    params = _axial_params(faces, axis)
    if not params:
        return None
    return min(params), max(params)


def _axially_adjacent(
    first_faces: Iterable[Face],
    second_faces: Iterable[Face],
    axis: Axis,
) -> bool:
    first_range = _axial_range(first_faces, axis)
    second_range = _axial_range(second_faces, axis)
    if first_range is None or second_range is None:
        return True
    gap = max(first_range[0], second_range[0]) - min(first_range[1], second_range[1])
    return gap <= _AXIS_GAP_TOL


def _angular_span(faces: Iterable[Face]) -> float:
    total = 0.0
    for face in faces:
        u_min, u_max, _v_min, _v_max = face._uv_bounds()
        total += abs(u_max - u_min)
    return total


def _cylinder_radius(face: Face, axis: Axis) -> float | None:
    if face.radius is not None:
        return face.radius
    # Trimmed cylinders often leave Face.radius unset; distance to axis is the radius.
    offset = (face.center() - axis.position).cross(axis.direction)
    return offset.length


def _canonical_direction(direction: Vector) -> Vector:
    unit = Vector(direction).normalized()
    for component in (unit.Z, unit.Y, unit.X):
        if abs(component) > TOLERANCE:
            return unit if component > 0 else -unit
    return unit


def _axial_center(faces: Iterable[Face], axis: Axis) -> Vector:
    params = _axial_params(faces, axis)
    if not params:
        return axis.position
    return axis.position + axis.direction * (0.5 * (min(params) + max(params)))


def _hole_from_group(axis: Axis, radius: float, faces: list[Face]) -> ProbedHole:
    direction = _canonical_direction(axis.direction)
    center = _axial_center(faces, Axis(axis.position, direction))
    return ProbedHole(center=center, axis=direction, diameter=2.0 * radius)
