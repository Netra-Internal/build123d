"""
build123d plates

name: plates.py
by:   Netra
date: September 9th 2026

desc:
    Pack parts onto 1..N printer plates and write a Bambu-style multi-plate
    3MF. Bambu CLI ``-arrange`` does not spill to new plates. This module
    assigns every part to a plate in Metadata/model_settings.config so a
    part cannot vanish off-bed. The library does not slice. Slicing stays
    with bambu_slice.py / Bambu CLI.

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

import json
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike, fsdecode
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from build123d.geometry import TOLERANCE
from build123d.mesher import Mesher
from build123d.topology.composite import Compound
from build123d.topology.shape_core import Shape

__all__ = [
    "A1_MINI_BED",
    "DEFAULT_EDGE_MARGIN",
    "DEFAULT_PART_GAP",
    "PartDoesNotFitError",
    "PlateAssignment",
    "PlatePackResult",
    "pack_plates",
    "inspect_plate_3mf",
]

A1_MINI_BED: tuple[float, float] = (180.0, 180.0)
DEFAULT_EDGE_MARGIN = 2.0
DEFAULT_PART_GAP = 8.0

_TESSELLATE_TOLERANCE = 0.001
_MODEL_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
_CONTENT_TYPES_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
_RELS_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_MODEL_REL_TYPE = "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"
_MODEL_CONTENT_TYPE = "application/vnd.ms-package.3dmanufacturing-3dmodel+xml"
_RELS_CONTENT_TYPE = "application/vnd.openxmlformats-package.relationships+xml"

_CONTENT_TYPES_NAME = "[Content_Types].xml"
_RELS_NAME = "_rels/.rels"
_MODEL_NAME = "3D/3dmodel.model"
_SETTINGS_NAME = "Metadata/model_settings.config"
_NETRA_NAME = "Metadata/netra_plates.json"


class PartDoesNotFitError(ValueError):
    """A single part's XY footprint exceeds the usable bed."""


@dataclass(frozen=True)
class PlateAssignment:
    """One part placed on one 1-based plate."""

    name: str
    plate: int
    origin_xy: tuple[float, float]
    footprint: tuple[float, float]
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready assignment."""
        return {
            "name": self.name,
            "plate": self.plate,
            "origin_xy": list(self.origin_xy),
            "footprint": list(self.footprint),
            "bbox_min": list(self.bbox_min),
            "bbox_max": list(self.bbox_max),
        }


@dataclass(frozen=True)
class PlatePackResult:
    """Plate assignments for every input part, plus an optional 3MF path."""

    bed: tuple[float, float]
    edge_margin: float
    part_gap: float
    usable: tuple[float, float]
    assignments: tuple[PlateAssignment, ...]
    plate_count: int
    path: str | None

    @property
    def ok(self) -> bool:
        """True iff every input part has exactly one assignment."""
        if not self.assignments:
            return False
        plates = {item.plate for item in self.assignments}
        return plates == set(range(1, self.plate_count + 1))

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready pack report."""
        return {
            "bed": list(self.bed),
            "edge_margin": self.edge_margin,
            "part_gap": self.part_gap,
            "usable": list(self.usable),
            "assignments": [item.to_dict() for item in self.assignments],
            "plate_count": self.plate_count,
            "path": self.path,
            "ok": self.ok,
        }


@dataclass(frozen=True)
class _LoadedPart:
    index: int
    name: str
    shape: Shape
    footprint: tuple[float, float]


def pack_plates(
    parts: Sequence[Shape | PathLike | str],
    *,
    bed: tuple[float, float] = A1_MINI_BED,
    edge_margin: float = DEFAULT_EDGE_MARGIN,
    part_gap: float = DEFAULT_PART_GAP,
    out: PathLike | str | bytes | None = None,
) -> PlatePackResult:
    """Pack parts onto 1..N plates and optionally write a multi-plate 3MF.

    Usable size is ``(bed[0] - 2*edge_margin, bed[1] - 2*edge_margin)``.
    The default physical bed is A1 mini 180 x 180. A part whose XY
    footprint exceeds usable raises ``PartDoesNotFitError`` and does not
    write ``out``. No part is dropped. Caller shapes are not mutated.

    The 3MF records plate assignments in ``Metadata/model_settings.config``
    and ``Metadata/netra_plates.json``. This function does not slice.

    Args:
        parts: Shapes or ``.stl`` paths, in input order.
        bed: Positive physical bed ``(x, y)`` in millimetres.
        edge_margin: Inset from each bed edge. Must be a number ``>= 0``.
        part_gap: Gap between parts on a shelf. Must be a number ``>= 0``.
        out: Destination 3MF path, or ``None`` to pack without writing.

    Returns:
        PlatePackResult: Assignments, usable size, plate count, and path.
    """
    items = _require_parts(parts)
    bed_value = _require_bed(bed)
    edge_value = _require_nonneg(edge_margin, "edge_margin")
    gap_value = _require_nonneg(part_gap, "part_gap")
    usable = (
        bed_value[0] - 2.0 * edge_value,
        bed_value[1] - 2.0 * edge_value,
    )
    loaded = [_load_one(part, index) for index, part in enumerate(items)]
    for part in loaded:
        if (
            part.footprint[0] > usable[0] + TOLERANCE
            or part.footprint[1] > usable[1] + TOLERANCE
        ):
            raise PartDoesNotFitError(
                f"{part.name} footprint {part.footprint} exceeds usable {usable}"
            )
    origins = _shelf_pack(
        loaded, bed=bed_value, edge_margin=edge_value, part_gap=gap_value
    )
    assignments: list[PlateAssignment] = []
    posed: list[tuple[_LoadedPart, PlateAssignment, Shape]] = []
    for part in loaded:
        plate, origin_xy = origins[part.index]
        placed = _pose(part.shape, origin_xy)
        bbox = placed.bounding_box()
        assignment = PlateAssignment(
            name=part.name,
            plate=plate,
            origin_xy=origin_xy,
            footprint=part.footprint,
            bbox_min=(float(bbox.min.X), float(bbox.min.Y), float(bbox.min.Z)),
            bbox_max=(float(bbox.max.X), float(bbox.max.Y), float(bbox.max.Z)),
        )
        assignments.append(assignment)
        posed.append((part, assignment, placed))
    path = fsdecode(out) if out is not None else None
    result = PlatePackResult(
        bed=bed_value,
        edge_margin=edge_value,
        part_gap=gap_value,
        usable=usable,
        assignments=tuple(assignments),
        plate_count=max(item.plate for item in assignments),
        path=path,
    )
    if path is not None:
        _write_3mf(path, posed, result)
    return result


def inspect_plate_3mf(path: PathLike | str | bytes) -> dict[str, Any]:
    """Read plate to object mapping from a packed 3MF zip.

    Requires ``Metadata/netra_plates.json`` and
    ``Metadata/model_settings.config``. Does not invoke Bambu Studio.
    """
    file_path = fsdecode(path)
    zip_path = Path(file_path)
    if not zip_path.is_file():
        raise FileNotFoundError(f"3MF file not found: {file_path}")
    with ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        if _NETRA_NAME not in names:
            raise ValueError(f"3MF missing {_NETRA_NAME}")
        if _SETTINGS_NAME not in names:
            raise ValueError(f"3MF missing {_SETTINGS_NAME}")
        netra = json.loads(archive.read(_NETRA_NAME).decode("utf-8"))
        settings_xml = archive.read(_SETTINGS_NAME)
    if not isinstance(netra, dict) or "assignments" not in netra:
        raise ValueError("netra_plates.json missing assignments")
    settings = _parse_model_settings(settings_xml)
    part_to_plate = {item["name"]: item["plate"] for item in netra["assignments"]}
    return {
        "netra_plates": netra,
        "model_settings": settings,
        "part_to_plate": part_to_plate,
        "object_to_plate": settings["object_to_plate"],
    }


def _require_parts(
    parts: Sequence[Shape | PathLike | str],
) -> list[Shape | PathLike | str]:
    # str is a Sequence of characters; a lone path must not be iterated.
    if isinstance(parts, (str, bytes)):
        raise ValueError("parts must be a non-empty sequence")
    try:
        items = list(parts)
    except TypeError as exc:
        raise ValueError("parts must be a non-empty sequence") from exc
    if not items:
        raise ValueError("parts must be a non-empty sequence")
    return items


def _require_nonneg(value: float, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number >= 0")
    if value < 0:
        raise ValueError(f"{name} must be a number >= 0")
    return float(value)


def _require_bed(bed: tuple[float, float]) -> tuple[float, float]:
    if not isinstance(bed, (tuple, list)) or len(bed) != 2:
        raise ValueError("bed must be two positive numbers")
    values: list[float] = []
    for value in bed:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("bed must be two positive numbers")
        values.append(float(value))
    return (values[0], values[1])


def _footprint(shape: Shape) -> tuple[float, float]:
    size = shape.bounding_box().size
    return (float(size.X), float(size.Y))


def _load_one(part: Shape | PathLike | str, index: int) -> _LoadedPart:
    if isinstance(part, Shape):
        name = part.label if part.label else f"part[{index}]"
        return _LoadedPart(
            index=index, name=name, shape=part, footprint=_footprint(part)
        )
    path = Path(fsdecode(part))
    if not path.is_file():
        raise FileNotFoundError(f"STL file not found: {path}")
    suffix = path.suffix.lower()
    if suffix != ".stl":
        raise ValueError(f"unknown suffix {path.suffix!r}")
    shapes = Mesher().read(path)
    if not shapes:
        raise ValueError(f"STL contained no mesh: {path}")
    shape = shapes[0] if len(shapes) == 1 else Compound(shapes)
    if not shape.label:
        shape.label = path.stem
    return _LoadedPart(
        index=index, name=shape.label, shape=shape, footprint=_footprint(shape)
    )


def _pose(shape: Shape, origin_xy: tuple[float, float]) -> Shape:
    bbox = shape.bounding_box()
    return shape.translate(
        (
            origin_xy[0] - float(bbox.min.X),
            origin_xy[1] - float(bbox.min.Y),
            -float(bbox.min.Z),
        )
    )


def _shelf_pack(
    parts: Sequence[_LoadedPart],
    *,
    bed: tuple[float, float],
    edge_margin: float,
    part_gap: float,
) -> dict[int, tuple[int, tuple[float, float]]]:
    right = bed[0] - edge_margin
    bottom = bed[1] - edge_margin
    plate = 1
    cursor_x = edge_margin
    cursor_y = edge_margin
    shelf_h = 0.0
    placed: dict[int, tuple[int, tuple[float, float]]] = {}
    ordered = sorted(
        parts,
        key=lambda part: (
            -(part.footprint[0] * part.footprint[1]),
            -max(part.footprint),
            part.index,
        ),
    )
    for part in ordered:
        width, depth = part.footprint
        if (
            cursor_x + width > right + TOLERANCE
            or cursor_y + depth > bottom + TOLERANCE
        ):
            next_y = cursor_y + shelf_h + part_gap
            if next_y + depth > bottom + TOLERANCE:
                plate += 1
                cursor_x = edge_margin
                cursor_y = edge_margin
                shelf_h = 0.0
            else:
                cursor_x = edge_margin
                cursor_y = next_y
                shelf_h = 0.0
        placed[part.index] = (plate, (cursor_x, cursor_y))
        cursor_x = cursor_x + width + part_gap
        shelf_h = max(shelf_h, depth)
    return placed


def _fmt(value: float) -> str:
    return f"{float(value):.9g}"


def _xml_bytes(root: ET.Element) -> bytes:
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _content_types_xml() -> bytes:
    types = ET.Element("Types", {"xmlns": _CONTENT_TYPES_NS})
    ET.SubElement(
        types,
        "Default",
        {"Extension": "rels", "ContentType": _RELS_CONTENT_TYPE},
    )
    ET.SubElement(
        types,
        "Default",
        {"Extension": "model", "ContentType": _MODEL_CONTENT_TYPE},
    )
    ET.SubElement(
        types, "Default", {"Extension": "json", "ContentType": "application/json"}
    )
    ET.SubElement(
        types, "Default", {"Extension": "config", "ContentType": "application/xml"}
    )
    return _xml_bytes(types)


def _rels_xml() -> bytes:
    rels = ET.Element("Relationships", {"xmlns": _RELS_NS})
    ET.SubElement(
        rels,
        "Relationship",
        {
            "Id": "rel-1",
            "Type": _MODEL_REL_TYPE,
            "Target": "/3D/3dmodel.model",
        },
    )
    return _xml_bytes(rels)


def _model_xml(posed: Sequence[tuple[_LoadedPart, PlateAssignment, Shape]]) -> bytes:
    model = ET.Element("model", {"unit": "millimeter", "xmlns": _MODEL_NS})
    resources = ET.SubElement(model, "resources")
    for part, _assignment, shape in posed:
        vertices, triangles = shape.tessellate(_TESSELLATE_TOLERANCE)
        obj = ET.SubElement(
            resources,
            "object",
            {
                "id": str(part.index + 1),
                "type": "model",
                "name": part.name,
            },
        )
        mesh = ET.SubElement(obj, "mesh")
        verts_el = ET.SubElement(mesh, "vertices")
        for vertex in vertices:
            ET.SubElement(
                verts_el,
                "vertex",
                {"x": _fmt(vertex.X), "y": _fmt(vertex.Y), "z": _fmt(vertex.Z)},
            )
        tris_el = ET.SubElement(mesh, "triangles")
        for triangle in triangles:
            ET.SubElement(
                tris_el,
                "triangle",
                {
                    "v1": str(triangle[0]),
                    "v2": str(triangle[1]),
                    "v3": str(triangle[2]),
                },
            )
    build = ET.SubElement(model, "build")
    for part, _assignment, _shape in posed:
        ET.SubElement(build, "item", {"objectid": str(part.index + 1)})
    return _xml_bytes(model)


def _model_settings_xml(
    posed: Sequence[tuple[_LoadedPart, PlateAssignment, Shape]],
) -> bytes:
    config = ET.Element("config")
    by_plate: dict[int, list[int]] = {}
    for part, assignment, _shape in posed:
        object_id = part.index + 1
        obj = ET.SubElement(config, "object", {"id": str(object_id)})
        ET.SubElement(obj, "metadata", {"key": "name", "value": part.name})
        by_plate.setdefault(assignment.plate, []).append(object_id)
    for plate in sorted(by_plate):
        plate_el = ET.SubElement(config, "plate")
        ET.SubElement(plate_el, "metadata", {"key": "plater_id", "value": str(plate)})
        for object_id in by_plate[plate]:
            instance = ET.SubElement(plate_el, "model_instance")
            ET.SubElement(
                instance, "metadata", {"key": "object_id", "value": str(object_id)}
            )
            ET.SubElement(instance, "metadata", {"key": "instance_id", "value": "0"})
    return _xml_bytes(config)


def _write_3mf(
    path: str,
    posed: Sequence[tuple[_LoadedPart, PlateAssignment, Shape]],
    result: PlatePackResult,
) -> None:
    payload = json.dumps(result.to_dict(), indent=2) + "\n"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(_CONTENT_TYPES_NAME, _content_types_xml())
        archive.writestr(_RELS_NAME, _rels_xml())
        archive.writestr(_MODEL_NAME, _model_xml(posed))
        archive.writestr(_SETTINGS_NAME, _model_settings_xml(posed))
        archive.writestr(_NETRA_NAME, payload.encode("utf-8"))


def _parse_model_settings(raw: bytes) -> dict[str, Any]:
    root = ET.fromstring(raw)
    objects: list[dict[str, Any]] = []
    for obj in root.findall("object"):
        object_id = int(obj.get("id", "0"))
        name = ""
        for meta in obj.findall("metadata"):
            if meta.get("key") == "name":
                name = meta.get("value") or ""
        objects.append({"id": object_id, "name": name})
    plates: list[dict[str, Any]] = []
    object_to_plate: dict[int, int] = {}
    for plate_el in root.findall("plate"):
        plater_id = 0
        for meta in plate_el.findall("metadata"):
            if meta.get("key") == "plater_id":
                plater_id = int(meta.get("value", "0"))
        object_ids: list[int] = []
        for instance in plate_el.findall("model_instance"):
            for meta in instance.findall("metadata"):
                if meta.get("key") != "object_id":
                    continue
                object_id = int(meta.get("value", "0"))
                if object_id in object_to_plate:
                    raise ValueError(
                        f"object {object_id} appears on more than one plate"
                    )
                object_to_plate[object_id] = plater_id
                object_ids.append(object_id)
        plates.append({"plater_id": plater_id, "object_ids": object_ids})
    return {
        "objects": objects,
        "plates": plates,
        "object_to_plate": object_to_plate,
    }
