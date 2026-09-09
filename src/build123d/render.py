"""
build123d render

name: render.py
by:   Netra
date: September 9th 2026

desc:
    Headless stills of Shape assemblies. Replaces a Fusion or Onshape
    screenshot glance. Supports view presets, per-body color including
    alpha, and a plane section-cut. No GUI. Meshes with Mesher and
    rasterizes with numpy. Pillow, already pulled in by
    threejs-materials, writes PNG and other stills.

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

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum, unique
from os import PathLike, fsdecode
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from build123d.build_enums import Keep
from build123d.geometry import Color, ColorLike, Plane, Vector, VectorLike
from build123d.mesher import Mesher
from build123d.topology.composite import Compound
from build123d.topology.shape_core import Shape

__all__ = [
    "ViewPreset",
    "SectionCut",
    "render",
]

_DEFAULT_RGBA = (0.85, 0.65, 0.2, 1.0)
_LINEAR_DEFLECTION = 0.1
_ANGULAR_DEFLECTION = 0.5
_FRAME_PAD = 0.08
_AMBIENT = 0.22
_DIFFUSE = 0.78


@unique
class ViewPreset(Enum):
    """Named camera directions. ``iso`` matches ``Drawing`` look_from=(1, -1, 1)."""

    ISO = "iso"
    FRONT = "front"
    BACK = "back"
    TOP = "top"
    BOTTOM = "bottom"
    RIGHT = "right"
    LEFT = "left"


@dataclass(frozen=True)
class SectionCut:
    """Keep one side of a plane so internals show.

    ``section()`` is the 2D intersection sketch. This is ``split``.
    """

    plane: Plane
    keep: Literal[Keep.TOP, Keep.BOTTOM] = Keep.TOP

    def __post_init__(self) -> None:
        if self.keep not in (Keep.TOP, Keep.BOTTOM):
            raise ValueError("section keep must be Keep.TOP or Keep.BOTTOM")


def render(
    shapes: Shape | Iterable[Shape],
    *,
    view: ViewPreset | str = ViewPreset.ISO,
    out: PathLike | str | bytes,
    size: tuple[int, int] = (800, 600),
    appearances: Mapping[Shape, ColorLike] | None = None,
    section: Plane | SectionCut | None = None,
    background: ColorLike = (1.0, 1.0, 1.0),
    look_at: VectorLike | None = None,
) -> Path:
    """Write a headless still of one or more shapes.

    Args:
        shapes: A Shape, Compound assembly, or iterable of those.
        view: Preset name (``iso``, ``front``, ``back``, ``top``,
            ``bottom``, ``right``, ``left``) or a ``ViewPreset``.
        out: Destination path. Suffix selects the still format (``.png``,
            ``.jpg``, ``.webp``, and others).
        size: ``(width, height)`` in pixels.
        appearances: Optional per-shape color override (object identity).
            Unlisted shapes use ``Shape.color`` or a default gold.
            Alpha < 1 is translucent.
        section: Plane or ``SectionCut`` that bisects each body and keeps
            one side. The camera stays on the uncut AABB.
        background: Clear color. Alpha is ignored; the still is RGB.
        look_at: Camera target. Defaults to the uncut bounding-box
            center. A section-cut keeps that frame.

    Returns:
        Path to the written file.

    Raises:
        ValueError: empty input, unknown view, invalid size, or nothing
            left after the section-cut.
    """
    width, height = size
    if width < 1 or height < 1:
        raise ValueError("size must be a pair of positive pixel counts")

    items = [shapes] if isinstance(shapes, Shape) else list(shapes)
    if not items:
        raise ValueError("nothing to render")

    uncut = _colored_bodies(items, appearances)
    if not uncut:
        raise ValueError("nothing to render")

    look_from, look_up = _view_basis(view)
    low, high = _world_aabb(shape for shape, _color in uncut)
    target = Vector(look_at) if look_at is not None else (low + high) * 0.5
    colored = _apply_cut(uncut, _as_section(section))
    if not colored:
        raise ValueError("nothing to render after section-cut")

    frame = _ortho_frame(low, high, look_from, look_up, target, (width, height))
    bg = tuple(Color(background))[:3]
    pixels = _rasterize(colored, (width, height), frame, bg)

    path = Path(fsdecode(out))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)
    return path


def _as_section(section: Plane | SectionCut | None) -> SectionCut | None:
    if section is None:
        return None
    if isinstance(section, SectionCut):
        return section
    return SectionCut(section)


_VIEW_BASIS: dict[ViewPreset, tuple[Vector, Vector]] = {
    ViewPreset.ISO: (Vector(1, -1, 1), Vector(0, 0, 1)),
    ViewPreset.FRONT: (Vector(0, -1, 0), Vector(0, 0, 1)),
    ViewPreset.BACK: (Vector(0, 1, 0), Vector(0, 0, 1)),
    ViewPreset.TOP: (Vector(0, 0, 1), Vector(0, 1, 0)),
    ViewPreset.BOTTOM: (Vector(0, 0, -1), Vector(0, 1, 0)),
    ViewPreset.RIGHT: (Vector(1, 0, 0), Vector(0, 0, 1)),
    ViewPreset.LEFT: (Vector(-1, 0, 0), Vector(0, 0, 1)),
}
if set(_VIEW_BASIS) != set(ViewPreset):
    missing = set(ViewPreset) - set(_VIEW_BASIS)
    raise RuntimeError(f"ViewPreset missing camera basis: {missing}")


def _view_basis(view: ViewPreset | str) -> tuple[Vector, Vector]:
    preset = view if isinstance(view, ViewPreset) else _preset_from_name(view)
    return _VIEW_BASIS[preset]


def _preset_from_name(name: str) -> ViewPreset:
    try:
        return ViewPreset(name.lower())
    except ValueError as exc:
        known = ", ".join(p.value for p in ViewPreset)
        raise ValueError(f"unknown view {name!r}; expected one of {known}") from exc


def _colored_bodies(
    items: list[Shape],
    appearances: Mapping[Shape, ColorLike] | None,
) -> list[tuple[Shape, Color]]:
    by_id = {id(shape): Color(color) for shape, color in (appearances or {}).items()}
    bodies: list[tuple[Shape, Color]] = []
    for item in items:
        bodies.extend(_collect_bodies(item, by_id, None))
    return bodies


def _collect_bodies(
    item: Shape,
    by_id: dict[int, Color],
    inherited: Color | None,
) -> list[tuple[Shape, Color]]:
    color = _color_of(item, by_id, inherited)
    children = item.children if isinstance(item, Compound) else []
    if children:
        collected: list[tuple[Shape, Color]] = []
        for child in children:
            collected.extend(_collect_bodies(child, by_id, color))
        if collected:
            return collected
    bodies: list[tuple[Shape, Color]] = []
    for body in _explode(item):
        piece_color = _color_of(body, by_id, color)
        bodies.append((body, piece_color))
    return bodies


def _apply_cut(
    colored: list[tuple[Shape, Color]], cut: SectionCut | None
) -> list[tuple[Shape, Color]]:
    if cut is None:
        return colored
    out: list[tuple[Shape, Color]] = []
    for shape, color in colored:
        out.extend((piece, color) for piece in _cut_body(shape, cut))
    return out


def _explode(shape: Shape) -> list[Shape]:
    solids = list(shape.solids())
    if solids:
        return solids
    if shape.faces():
        return [shape]
    return []


def _color_of(shape: Shape, by_id: dict[int, Color], inherited: Color | None) -> Color:
    if id(shape) in by_id:
        return by_id[id(shape)]
    if shape.color is not None:
        return shape.color
    if inherited is not None:
        return inherited
    return Color(*_DEFAULT_RGBA)


def _cut_body(body: Shape, cut: SectionCut | None) -> list[Shape]:
    if cut is None:
        return [body]
    result = body.split(cut.plane, keep=cut.keep)
    if result is None:
        return []
    if isinstance(result, list):
        return [piece for piece in result if piece is not None]
    return [result]


def _world_aabb(shapes: Iterable[Shape]) -> tuple[Vector, Vector]:
    mins = []
    maxs = []
    for shape in shapes:
        box = shape.bounding_box()
        mins.append(box.min)
        maxs.append(box.max)
    low = Vector(
        min(p.X for p in mins),
        min(p.Y for p in mins),
        min(p.Z for p in mins),
    )
    high = Vector(
        max(p.X for p in maxs),
        max(p.Y for p in maxs),
        max(p.Z for p in maxs),
    )
    return low, high


@dataclass(frozen=True)
class _OrthoFrame:
    origin: np.ndarray
    mid: np.ndarray
    scale: float
    right: np.ndarray
    up: np.ndarray
    forward: np.ndarray


def _ortho_frame(
    low: Vector,
    high: Vector,
    look_from: Vector,
    look_up: Vector,
    target: Vector,
    size: tuple[int, int],
) -> _OrthoFrame:
    width, height = size
    right, up, forward = _camera_basis(look_from, look_up)
    origin = _vec3(target)
    corners = np.array(
        [
            (x, y, z)
            for x in (low.X, high.X)
            for y in (low.Y, high.Y)
            for z in (low.Z, high.Z)
        ],
        dtype=np.float64,
    )
    cam_xy = _to_camera(corners, origin, right, up, forward)[:, :2]
    low_xy = cam_xy.min(axis=0)
    high_xy = cam_xy.max(axis=0)
    span = np.maximum(high_xy - low_xy, 1e-9)
    scale = (1.0 - 2.0 * _FRAME_PAD) * min(width, height) / float(span.max())
    mid = (low_xy + high_xy) * 0.5
    return _OrthoFrame(origin, mid, scale, right, up, forward)


def _mesh(shape: Shape) -> tuple[np.ndarray, np.ndarray]:
    vertices, triangles = Mesher._mesh_shape(
        shape, _LINEAR_DEFLECTION, _ANGULAR_DEFLECTION
    )
    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(triangles, dtype=np.int32),
    )


def _vec3(vector: Vector) -> np.ndarray:
    return np.array((vector.X, vector.Y, vector.Z), dtype=np.float64)


def _camera_basis(
    look_from: Vector, look_up: Vector
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = -_vec3(look_from.normalized())
    up = _vec3(look_up.normalized())
    right = np.cross(forward, up)
    norm = np.linalg.norm(right)
    if norm < 1e-12:
        raise ValueError("look_up is parallel to the view direction")
    right = right / norm
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    return right, up, forward


def _to_camera(
    points: np.ndarray,
    origin: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    forward: np.ndarray,
) -> np.ndarray:
    rel = points - origin
    return np.column_stack((rel @ right, rel @ up, rel @ forward))


def _rasterize(
    colored: list[tuple[Shape, Color]],
    size: tuple[int, int],
    frame: _OrthoFrame,
    background: tuple[float, float, float],
) -> np.ndarray:
    width, height = size
    origin = frame.origin
    mid = frame.mid
    scale = frame.scale
    right, up, forward = frame.right, frame.up, frame.forward

    meshes: list[tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]] = []
    for shape, color in colored:
        verts, tris = _mesh(shape)
        if verts.size == 0 or tris.size == 0:
            continue
        meshes.append((verts, tris, tuple(color)))
    if not meshes:
        raise ValueError("nothing to render: no triangulated faces")

    def to_screen(verts: np.ndarray) -> np.ndarray:
        cam = _to_camera(verts, origin, right, up, forward)
        sx = (cam[:, 0] - mid[0]) * scale + width / 2.0
        sy = height / 2.0 - (cam[:, 1] - mid[1]) * scale
        return np.column_stack((sx, sy, cam[:, 2]))

    image = np.empty((height, width, 3), dtype=np.float64)
    image[:] = background
    depth = np.full((height, width), np.inf, dtype=np.float64)

    opaque: list[tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]] = []
    glass: list[tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]] = []
    for verts, tris, rgba in meshes:
        (glass if rgba[3] < 0.999 else opaque).append((verts, tris, rgba))

    for verts, tris, rgba in opaque:
        _stamp(
            image, depth, to_screen(verts), verts, tris, rgba, forward, write_depth=True
        )

    def farthest_z(
        item: tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]],
    ) -> float:
        cam = _to_camera(item[0], origin, right, up, forward)
        return float(cam[:, 2].mean())

    for verts, tris, rgba in sorted(glass, key=farthest_z):
        _stamp(
            image,
            depth,
            to_screen(verts),
            verts,
            tris,
            rgba,
            forward,
            write_depth=False,
        )

    return (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def _stamp(
    image: np.ndarray,
    depth: np.ndarray,
    screen: np.ndarray,
    verts: np.ndarray,
    tris: np.ndarray,
    rgba: tuple[float, float, float, float],
    light: np.ndarray,
    *,
    write_depth: bool,
) -> None:
    height, width = depth.shape
    red, green, blue, alpha = rgba
    for tri in tris:
        a, b, c = screen[tri]
        area = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
        if abs(area) < 1e-9:
            continue
        n = np.cross(verts[tri[1]] - verts[tri[0]], verts[tri[2]] - verts[tri[0]])
        nlen = np.linalg.norm(n)
        if nlen < 1e-12:
            continue
        shade = _AMBIENT + _DIFFUSE * max(0.0, float(abs((n / nlen) @ light)))
        rgb = np.array((red, green, blue), dtype=np.float64) * shade
        minx = max(int(np.floor(min(a[0], b[0], c[0]))), 0)
        maxx = min(int(np.ceil(max(a[0], b[0], c[0]))), width - 1)
        miny = max(int(np.floor(min(a[1], b[1], c[1]))), 0)
        maxy = min(int(np.ceil(max(a[1], b[1], c[1]))), height - 1)
        if minx > maxx or miny > maxy:
            continue
        xs = np.arange(minx, maxx + 1)
        ys = np.arange(miny, maxy + 1)
        xx, yy = np.meshgrid(xs, ys)
        w0 = ((b[0] - a[0]) * (yy - a[1]) - (b[1] - a[1]) * (xx - a[0])) / area
        w1 = ((c[0] - b[0]) * (yy - b[1]) - (c[1] - b[1]) * (xx - b[0])) / area
        w2 = ((a[0] - c[0]) * (yy - c[1]) - (a[1] - c[1]) * (xx - c[0])) / area
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        z = w0 * a[2] + w1 * b[2] + w2 * c[2]
        sl = np.s_[miny : maxy + 1, minx : maxx + 1]
        visible = inside & (z <= depth[sl])
        if not visible.any():
            continue
        dst = image[sl]
        if write_depth:
            dst[visible] = rgb
            depth[sl][visible] = z[visible]
        else:
            dst[visible] = dst[visible] * (1.0 - alpha) + rgb * alpha
