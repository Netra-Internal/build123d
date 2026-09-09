"""
build123d viewer

name: viewer.py
by:   Netra
date: September 9th 2026

desc:
    Live HTML handoff of Shape assemblies. One self-contained file with
    embedded glTF JSON and three.js OrbitControls. Reuses render's body
    walk, section-cut, and mesh. Does not call export_gltf.

license:

    Copyright 2026 The build123d Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/Apache-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from os import PathLike, fsdecode
from pathlib import Path
from typing import Any

import numpy as np

from build123d.geometry import Color, ColorLike, Plane
from build123d.render import (
    SectionCut,
    _apply_cut,
    _as_section,
    _colored_bodies,
    _mesh,
)
from build123d.topology.shape_core import Shape

# pylint: disable=missing-class-docstring

__all__ = [
    "show",
]

_GLTF_FLOAT = 5126
_GLTF_UNSIGNED_INT = 5125
_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963
_GLASS_ALPHA = 0.999
_CAD_UP = (0.0, 0.0, 1.0)


@dataclass(frozen=True)
class ViewerPart:
    positions: bytes
    indices: bytes
    rgba: tuple[float, float, float, float]


@dataclass(frozen=True)
class ViewerDocument:
    parts: tuple[ViewerPart, ...]


def show(
    shapes: Shape | Iterable[Shape],
    *,
    out: PathLike | str | bytes,
    appearances: Mapping[Shape, ColorLike] | None = None,
    section: Plane | SectionCut | None = None,
) -> Path:
    """Write a live HTML viewer of one or more shapes.

    Args:
        shapes: A Shape, Compound assembly, or iterable of those.
        out: Destination path. The file is a single HTML document.
        appearances: Optional per-shape color override (object identity).
            Unlisted shapes use ``Shape.color`` or a default gold.
            Alpha < 1 is translucent.
        section: Plane or ``SectionCut`` that bisects each body and keeps
            one side. Same semantics as ``render``.

    Returns:
        Absolute path to the written file. Also printed.

    Raises:
        ValueError: empty input, or nothing left after the section-cut
            or meshing.
    """
    items = [shapes] if isinstance(shapes, Shape) else list(shapes)
    if not items:
        raise ValueError("nothing to show")

    uncut = _colored_bodies(items, appearances)
    if not uncut:
        raise ValueError("nothing to show")

    colored = _apply_cut(uncut, _as_section(section))
    if not colored:
        raise ValueError("nothing to show after section-cut")

    document = _document_from_bodies(colored)
    html = _html_from_gltf(_gltf_from_document(document))

    path = Path(fsdecode(out))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    resolved = path.resolve()
    print(resolved)
    return resolved


def _document_from_bodies(colored: list[tuple[Shape, Color]]) -> ViewerDocument:
    parts: list[ViewerPart] = []
    for shape, color in colored:
        verts, tris = _mesh(shape)
        if verts.size == 0 or tris.size == 0:
            continue
        positions = np.ascontiguousarray(verts, dtype="<f4").tobytes()
        indices = np.ascontiguousarray(tris, dtype="<u4").tobytes()
        red, green, blue, alpha = tuple(color)
        parts.append(
            ViewerPart(
                positions,
                indices,
                (float(red), float(green), float(blue), float(alpha)),
            )
        )
    if not parts:
        raise ValueError("nothing to show: no triangulated faces")
    return ViewerDocument(tuple(parts))


def _gltf_from_document(document: ViewerDocument) -> dict[str, Any]:
    blob = bytearray()
    buffer_views: list[dict[str, int]] = []
    accessors: list[dict[str, Any]] = []
    meshes: list[dict[str, Any]] = []
    materials: list[dict[str, Any]] = []
    nodes: list[dict[str, int]] = []

    for part in document.parts:
        pos_view, pos_acc = _append_positions(blob, buffer_views, part.positions)
        idx_view, idx_acc = _append_indices(blob, buffer_views, part.indices)
        accessors.append(pos_acc | {"bufferView": pos_view})
        accessors.append(idx_acc | {"bufferView": idx_view})
        position_index = len(accessors) - 2
        index_index = len(accessors) - 1
        material_index = len(materials)
        materials.append(_material(part.rgba))
        meshes.append(
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": position_index},
                        "indices": index_index,
                        "material": material_index,
                    }
                ]
            }
        )
        nodes.append({"mesh": len(meshes) - 1})

    payload = bytes(blob)
    uri = "data:application/octet-stream;base64," + base64.b64encode(payload).decode(
        "ascii"
    )
    return {
        "asset": {"version": "2.0", "generator": "build123d.show"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(payload), "uri": uri}],
    }


def _append_positions(
    blob: bytearray,
    views: list[dict[str, int]],
    positions: bytes,
) -> tuple[int, dict[str, Any]]:
    offset = len(blob)
    blob.extend(positions)
    views.append(
        {
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(positions),
            "target": _ARRAY_BUFFER,
        }
    )
    xyz = np.frombuffer(positions, dtype="<f4").reshape(-1, 3)
    return len(views) - 1, {
        "componentType": _GLTF_FLOAT,
        "count": int(xyz.shape[0]),
        "type": "VEC3",
        "min": [float(v) for v in xyz.min(axis=0)],
        "max": [float(v) for v in xyz.max(axis=0)],
    }


def _append_indices(
    blob: bytearray,
    views: list[dict[str, int]],
    indices: bytes,
) -> tuple[int, dict[str, Any]]:
    offset = len(blob)
    blob.extend(indices)
    views.append(
        {
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(indices),
            "target": _ELEMENT_ARRAY_BUFFER,
        }
    )
    return len(views) - 1, {
        "componentType": _GLTF_UNSIGNED_INT,
        "count": len(indices) // 4,
        "type": "SCALAR",
    }


def _material(rgba: tuple[float, float, float, float]) -> dict[str, Any]:
    red, green, blue, alpha = rgba
    material: dict[str, Any] = {
        "pbrMetallicRoughness": {
            "baseColorFactor": [red, green, blue, alpha],
            "metallicFactor": 0.0,
            "roughnessFactor": 0.7,
        },
        "doubleSided": True,
    }
    if alpha < _GLASS_ALPHA:
        material["alphaMode"] = "BLEND"
    return material


def _html_from_gltf(gltf: dict[str, Any]) -> str:
    payload = json.dumps(gltf, separators=(",", ":"), ensure_ascii=True)
    payload = payload.replace("<", r"\u003c")
    up = ", ".join(str(axis) for axis in _CAD_UP)
    return _HTML.replace("__NETRA_GLTF__", payload).replace("__CAD_UP__", up)


_HTML = """<!DOCTYPE html>
<html lang="en" data-engine="three.js">
<head>
<meta charset="utf-8">
<title>build123d viewer</title>
<style>
html,body{margin:0;height:100%;overflow:hidden;background:#f4f4f4}
canvas{display:block;width:100%;height:100%}
</style>
<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/"
  }
}
</script>
</head>
<body>
<script type="application/json" id="netra-gltf">__NETRA_GLTF__</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

window.addEventListener("unhandledrejection", (event) => {
  document.body.dataset.viewerError = String(event.reason);
});

const spec = document.getElementById("netra-gltf").textContent;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf4f4f4);
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
camera.up.set(__CAD_UP__);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio || 1);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const key = new THREE.DirectionalLight(0xffffff, 0.85);
key.position.set(1, -1, 2);
scene.add(key);

function resize() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
}
window.addEventListener("resize", resize);
resize();

function fit(root) {
  const box = new THREE.Box3().setFromObject(root);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);
  const radius = 0.5 * Math.max(size.x, size.y, size.z, 1e-6);
  const dist = (radius / Math.tan((camera.fov * Math.PI) / 360)) * 1.35;
  camera.position.set(center.x + dist, center.y - dist, center.z + dist);
  camera.near = Math.max(dist / 200, 0.01);
  camera.far = dist * 100;
  camera.updateProjectionMatrix();
  camera.lookAt(center);
  controls.target.copy(center);
  controls.update();
}

function glass(root) {
  root.traverse((obj) => {
    if (!obj.isMesh) return;
    const list = Array.isArray(obj.material) ? obj.material : [obj.material];
    for (const mat of list) {
      if ((mat.opacity ?? 1) < 0.999) {
        mat.transparent = true;
        mat.depthWrite = false;
      }
    }
  });
}

new GLTFLoader().parse(
  spec,
  "",
  (gltf) => {
    const root = gltf.scene;
    glass(root);
    scene.add(root);
    fit(root);
    document.body.dataset.viewerReady = "1";
  },
  (err) => {
    document.body.dataset.viewerError = String(err);
  }
);

function tick() {
  requestAnimationFrame(tick);
  controls.update();
  renderer.render(scene, camera);
}
tick();
</script>
</body>
</html>
"""
