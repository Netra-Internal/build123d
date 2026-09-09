"""
build123d print

name: print.py
by:   Netra
date: September 9th 2026

desc:
    Library-first printability report and bed-oriented STL export. Overhangs
    use face normals, walls use inward offset collapse, and islands are
    disconnected solids above the bed. Pose a chosen face onto the bed
    before writing STL.

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
from math import asin, degrees
from os import PathLike, fsdecode
from typing import Any

from build123d.exporters3d import export_stl
from build123d.geometry import TOLERANCE, Axis, Location, Plane, Vector, VectorLike
from build123d.operations_generic import offset
from build123d.topology.shape_core import Shape
from build123d.topology.two_d import Face

__all__ = [
    "OverhangRecord",
    "WallRecord",
    "IslandRecord",
    "PrintCheckResult",
    "print_check",
    "PrintExportResult",
    "export_for_print",
]

_WALL_METHOD = "offset_collapse"
_WALL_SEARCH_ITERS = 48


@dataclass(frozen=True)
class OverhangRecord:
    """One face steeper than the overhang threshold, not on the bed."""

    area: float
    angle_deg: float
    center: Vector
    normal: Vector

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready overhang record."""
        return {
            "area": self.area,
            "angle_deg": self.angle_deg,
            "center": _xyz(self.center),
            "normal": _xyz(self.normal),
        }


@dataclass(frozen=True)
class WallRecord:
    """Inward-offset collapse estimate against ``min_wall``."""

    min_wall: float
    measured: float | None
    ok: bool
    method: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready wall record."""
        return {
            "min_wall": self.min_wall,
            "measured": self.measured,
            "ok": self.ok,
            "method": self.method,
        }


@dataclass(frozen=True)
class IslandRecord:
    """A disconnected solid whose bbox sits above the bed."""

    name: str
    bbox_min: Vector
    bbox_max: Vector

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready island record."""
        return {
            "name": self.name,
            "bbox_min": _xyz(self.bbox_min),
            "bbox_max": _xyz(self.bbox_max),
        }


@dataclass(frozen=True)
class PrintCheckResult:
    """Overhang, wall, and island report for one part."""

    overhang_deg: float
    min_wall: float
    overhangs: tuple[OverhangRecord, ...]
    wall: WallRecord
    islands: tuple[IslandRecord, ...]

    @property
    def ok(self) -> bool:
        """True when no flagged overhangs, wall.ok, and no islands."""
        return not self.overhangs and self.wall.ok and not self.islands

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready printability report."""
        return {
            "overhang_deg": self.overhang_deg,
            "min_wall": self.min_wall,
            "ok": self.ok,
            "overhangs": [record.to_dict() for record in self.overhangs],
            "wall": self.wall.to_dict(),
            "islands": [record.to_dict() for record in self.islands],
        }


def print_check(
    part: Shape,
    *,
    overhang_deg: float = 45.0,
    min_wall: float = 0.8,
    print_up: VectorLike = (0, 0, 1),
) -> PrintCheckResult:
    """Flag overhangs, thin walls, and disconnected solids before you slice.

    Overhangs use each face normal against ``print_up``. The angle is
    measured from vertical, so a wall is 0° and a downward face is 90°.
    A face flags when that angle is strictly greater than ``overhang_deg``.
    A 45° face at the default 45° threshold is OK. Faces on the bed
    (vertices at the part minimum along ``print_up``) are not overhangs.

    Wall thickness is an inward-offset collapse heuristic, not a
    medial-axis or slicer wall measure. ``offset(solid, -min_wall / 2)``
    must keep positive volume.

    Islands are disconnected solids whose bbox minimum sits above the
    bed. A connected mushroom is one solid, so it is not an island.
    This is not a per-layer slicer island check.

    The library does not raise when a check fails. Read ``ok``.

    Args:
        part: Solid or compound to inspect.
        overhang_deg: Flag faces steeper than this angle from vertical.
            Must be a number ``>= 0``.
        min_wall: Inward-offset collapse threshold. Must be a number
            ``>= 0``.
        print_up: Build direction. Must be a non-zero vector.

    Returns:
        PrintCheckResult: Frozen report with ``overhangs``, ``wall``,
        and ``islands``.
    """
    overhang_value = _require_nonneg(overhang_deg, "overhang_deg")
    wall_value = _require_nonneg(min_wall, "min_wall")
    up = _require_print_up(print_up)
    overhangs = _overhangs(part, overhang_value, up)
    solids = list(part.solids())
    wall = _wall_record(solids, wall_value)
    islands = _islands(solids, up)
    return PrintCheckResult(
        overhang_deg=overhang_value,
        min_wall=wall_value,
        overhangs=overhangs,
        wall=wall,
        islands=islands,
    )


@dataclass(frozen=True)
class PrintExportResult:
    """STL written after posing ``face_down`` onto the bed."""

    path: str
    size: tuple[float, float, float]
    bed: tuple[float, float, float]

    @property
    def ok(self) -> bool:
        """True only on success (the function raises otherwise)."""
        return True

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready export record."""
        return {
            "path": self.path,
            "size": list(self.size),
            "bed": list(self.bed),
            "ok": self.ok,
        }


def export_for_print(
    part: Shape,
    *,
    face_down: Face,
    bed: tuple[float, float, float] = (180.0, 180.0, 180.0),
    out: PathLike | str | bytes,
) -> PrintExportResult:
    """Pose ``face_down`` onto the bed and write an STL.

    Builds ``Plane(origin=face_down.center(), z_dir=face_down.normal_at())``,
    moves the part into ``Plane.XY``, rotates 180° about ``Axis.X``, then
    translates so ``bounding_box().min.Z == 0``. The input ``part`` is
    left unchanged.

    Raises ``ValueError`` and does not write ``out`` when the posed AABB
    exceeds ``bed`` along X, Y, or Z.

    Args:
        part: Solid or compound to export.
        face_down: Face that should sit on the bed.
        bed: Positive ``(x, y, z)`` build volume.
        out: Destination STL path.

    Returns:
        PrintExportResult: Path, posed AABB size, and bed.
    """
    if not isinstance(face_down, Face):
        raise ValueError("face_down must be a Face")
    bed_value = _require_bed(bed)
    posed = _pose_face_down(part, face_down)
    size_vec = posed.bounding_box().size
    size = (float(size_vec.X), float(size_vec.Y), float(size_vec.Z))
    if (
        size[0] > bed_value[0] + TOLERANCE
        or size[1] > bed_value[1] + TOLERANCE
        or size[2] > bed_value[2] + TOLERANCE
    ):
        raise ValueError(f"posed size {size} exceeds bed {bed_value}")
    path = fsdecode(out)
    export_stl(posed, out)
    return PrintExportResult(path=path, size=size, bed=bed_value)


def _xyz(vector: Vector) -> list[float]:
    return [vector.X, vector.Y, vector.Z]


def _require_nonneg(value: float, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number >= 0")
    if value < 0:
        raise ValueError(f"{name} must be a number >= 0")
    return float(value)


def _require_print_up(print_up: VectorLike) -> Vector:
    vec = Vector(print_up)
    if vec.length == 0:
        raise ValueError("print_up must be a non-zero vector")
    return vec.normalized()


def _require_bed(bed: tuple[float, float, float]) -> tuple[float, float, float]:
    if not isinstance(bed, (tuple, list)) or len(bed) != 3:
        raise ValueError("bed must be three positive numbers")
    values: list[float] = []
    for value in bed:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("bed must be three positive numbers")
        values.append(float(value))
    return (values[0], values[1], values[2])


def _min_vertex_projection(shape: Shape, up: Vector) -> float | None:
    vertices = shape.vertices()
    if not vertices:
        return None
    return min(Vector(vertex).dot(up) for vertex in vertices)


def _overhangs(
    part: Shape, overhang_deg: float, up: Vector
) -> tuple[OverhangRecord, ...]:
    bed_height = _min_vertex_projection(part, up)
    records: list[OverhangRecord] = []
    for face in part.faces():
        normal = face.normal_at()
        theta = degrees(asin(min(1.0, max(0.0, -normal.dot(up)))))
        if theta <= overhang_deg:
            continue
        if bed_height is not None:
            face_height = _min_vertex_projection(face, up)
            if face_height is not None and abs(face_height - bed_height) <= TOLERANCE:
                continue
        records.append(
            OverhangRecord(
                area=face.area,
                angle_deg=theta,
                center=face.center(),
                normal=normal,
            )
        )
    records.sort(
        key=lambda record: (
            -record.angle_deg,
            -record.area,
            record.center.X,
            record.center.Y,
            record.center.Z,
        )
    )
    return tuple(records)


def _offset_survives(solid: Shape, thickness: float) -> bool:
    # offset(solid, 0) raises on live OCCT; a zero inset is a no-op.
    if thickness <= 0:
        return True
    try:
        result = offset(solid, -thickness / 2.0)
    except (ValueError, RuntimeError):
        return False
    if result.is_null:
        return False
    try:
        return result.volume > TOLERANCE
    except ValueError:
        return False


def _measure_wall(solid: Shape) -> float | None:
    cap = solid.bounding_box().diagonal
    if cap <= TOLERANCE:
        return None
    lo = 0.0
    hi = cap
    if _offset_survives(solid, hi):
        return hi
    for _ in range(_WALL_SEARCH_ITERS):
        mid = (lo + hi) / 2.0
        if _offset_survives(solid, mid):
            lo = mid
        else:
            hi = mid
    if lo <= TOLERANCE:
        return None
    return lo


def _wall_record(solids: list[Shape], min_wall: float) -> WallRecord:
    if not solids:
        return WallRecord(
            min_wall=min_wall, measured=None, ok=True, method=_WALL_METHOD
        )
    ok = True
    measured_values: list[float] = []
    for solid in solids:
        if not _offset_survives(solid, min_wall):
            ok = False
        measured = _measure_wall(solid)
        if measured is not None:
            measured_values.append(measured)
    return WallRecord(
        min_wall=min_wall,
        measured=min(measured_values) if measured_values else None,
        ok=ok,
        method=_WALL_METHOD,
    )


def _islands(solids: list[Shape], up: Vector) -> tuple[IslandRecord, ...]:
    if not solids:
        return ()
    heights = [solid.bounding_box().min.dot(up) for solid in solids]
    bed = min(heights)
    records: list[IslandRecord] = []
    for index, (solid, height) in enumerate(zip(solids, heights)):
        if height <= bed + TOLERANCE:
            continue
        bbox = solid.bounding_box()
        name = solid.label if solid.label else f"solid[{index}]"
        records.append(IslandRecord(name=name, bbox_min=bbox.min, bbox_max=bbox.max))
    records.sort(
        key=lambda record: (
            record.name,
            record.bbox_min.X,
            record.bbox_min.Y,
            record.bbox_min.Z,
        )
    )
    return tuple(records)


def _pose_face_down(part: Shape, face_down: Face) -> Shape:
    face_plane = Plane(origin=face_down.center(), z_dir=face_down.normal_at())
    posed = part.moved(Location(Plane.XY) * Location(face_plane).inverse())
    posed = posed.rotate(Axis.X, 180)
    return posed.translate((0, 0, -posed.bounding_box().min.Z))
