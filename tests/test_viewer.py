"""
build123d viewer tests

name: test_viewer.py
by:   Netra
date: September 9th 2026

desc: Live HTML handoff. Embedded glTF, color, and section-cut.
"""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from build123d import Box, Color, Compound, Plane, Pos, show


def _html_path(directory: str, name: str = "out.html") -> str:
    return os.path.join(directory, name)


def _embedded_gltf(html: str) -> dict:
    marker = html.find('id="netra-gltf"')
    if marker < 0:
        raise AssertionError("missing #netra-gltf")
    start = html.find(">", marker) + 1
    end = html.find("</script>", start)
    if start <= 0 or end < 0:
        raise AssertionError("unclosed #netra-gltf script")
    return json.loads(html[start:end])


def _write_show(directory: str, shapes, name: str = "out.html", **kwargs):
    buf = io.StringIO()
    with redirect_stdout(buf):
        path = show(shapes, out=_html_path(directory, name), **kwargs)
    return path, Path(path).read_text(encoding="utf-8"), buf.getvalue()


def _position_counts(gltf: dict) -> list[int]:
    counts = []
    for mesh in gltf["meshes"]:
        for prim in mesh["primitives"]:
            acc = gltf["accessors"][prim["attributes"]["POSITION"]]
            counts.append(acc["count"])
    return counts


def _base_colors(gltf: dict) -> list[list[float]]:
    return [
        material["pbrMetallicRoughness"]["baseColorFactor"]
        for material in gltf["materials"]
    ]


class TestViewerLibrary(unittest.TestCase):
    """HTML file bytes, embedded glTF, appearance, and section-cut."""

    def test_box_writes_absolute_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, text, _printed = _write_show(tmp, Box(10, 6, 4))

            self.assertTrue(path.is_absolute())
            self.assertTrue(path.exists())
            head = text.lstrip().lower()
            self.assertTrue(
                head.startswith("<!doctype html") or head.startswith("<html")
            )
            self.assertIn("three.js", text.lower())
            self.assertIn("gltf", text.lower())

    def test_embedded_gltf_is_version_2_with_mesh(self):
        with tempfile.TemporaryDirectory() as tmp:
            _path, text, _printed = _write_show(tmp, Box(10, 6, 4))
            gltf = _embedded_gltf(text)

        self.assertEqual(gltf["asset"]["version"], "2.0")
        self.assertEqual(_position_counts(gltf), [24])
        self.assertEqual(gltf["buffers"][0]["byteLength"], 432)

    def test_translucent_red_writes_base_color_factor(self):
        box = Box(8, 8, 8)
        box.color = Color(0.9, 0.05, 0.05, 0.25)

        with tempfile.TemporaryDirectory() as tmp:
            _path, text, _printed = _write_show(tmp, box)
            red, green, blue, alpha = _base_colors(_embedded_gltf(text))[0]

        self.assertGreater(red, green + 0.5)
        self.assertGreater(red, blue + 0.5)
        self.assertAlmostEqual(alpha, 0.25, delta=0.05)

    def test_appearances_mapping_overrides_shape_color(self):
        box = Box(8, 8, 8)
        box.color = Color(0.0, 1.0, 0.0)

        with tempfile.TemporaryDirectory() as tmp:
            _path, text, _printed = _write_show(
                tmp,
                box,
                appearances={box: Color(0.05, 0.1, 0.9, 1.0)},
            )
            red, green, blue, alpha = _base_colors(_embedded_gltf(text))[0]

        self.assertGreater(blue, red + 0.4)
        self.assertGreater(blue, green)
        self.assertAlmostEqual(alpha, 1.0, delta=0.05)

    def test_section_cut_changes_embedded_gltf(self):
        hollow = Box(20, 20, 20) - Box(12, 12, 12)

        with tempfile.TemporaryDirectory() as tmp:
            _uncut_path, uncut_html, _p1 = _write_show(tmp, hollow, name="uncut.html")
            _cut_path, cut_html, _p2 = _write_show(
                tmp, hollow, name="cut.html", section=Plane.YZ
            )
            uncut = _embedded_gltf(uncut_html)
            cut = _embedded_gltf(cut_html)

        self.assertTrue(
            _position_counts(uncut) != _position_counts(cut)
            or uncut["buffers"][0]["byteLength"] != cut["buffers"][0]["byteLength"]
        )

    def test_empty_input_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                show([], out=_html_path(tmp))
            with self.assertRaises(ValueError):
                show(Compound(), out=_html_path(tmp, "empty-compound.html"))

    def test_compound_children_keep_per_body_color(self):
        left = Box(6, 6, 2)
        left.color = Color(0.95, 0.05, 0.05, 1.0)
        right = Pos(10, 0, 0) * Box(6, 6, 2)
        right.color = Color(0.05, 0.05, 0.95, 1.0)
        assy = Compound(children=[left, right])

        with tempfile.TemporaryDirectory() as tmp:
            _path, text, _printed = _write_show(tmp, assy)
            colors = _base_colors(_embedded_gltf(text))

        self.assertEqual(len(colors), 2)
        reds = [c for c in colors if c[0] > c[2] + 0.4]
        navies = [c for c in colors if c[2] > c[0] + 0.4]
        self.assertEqual(len(reds), 1)
        self.assertEqual(len(navies), 1)

    def test_printed_path_is_resolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _text, printed = _write_show(tmp, Box(2, 2, 2))

        self.assertEqual(printed.strip(), str(path.resolve()))

    def test_section_that_misses_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            far = Plane.XY.offset(1000)
            with self.assertRaises(ValueError):
                show(Box(2, 2, 2), out=_html_path(tmp), section=far)

    def test_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "nested" / "dir" / "out.html"
            buf = io.StringIO()
            with redirect_stdout(buf):
                path = show(Box(3, 3, 3), out=dest)
            self.assertTrue(path.exists())
            self.assertTrue(path.is_absolute())
