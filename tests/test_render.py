"""
build123d render tests

name: test_render.py
by:   Netra
date: September 9th 2026

desc: Headless PNG render. File bytes, appearance, and section-cut.
"""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from build123d import (
    Box,
    Color,
    Compound,
    Keep,
    Plane,
    Pos,
    SectionCut,
    Vector,
    ViewPreset,
    render,
)
from build123d.render import _camera_basis


def _png_path(directory: str, name: str = "out.png") -> str:
    return os.path.join(directory, name)


def _mean_nonwhite_arr(pixels: np.ndarray) -> np.ndarray:
    mask = pixels.sum(axis=2) < 750
    if not mask.any():
        raise AssertionError("image has no non-white pixels")
    return pixels[mask].mean(axis=0)


def _mean_nonwhite(path: str) -> np.ndarray:
    return _mean_nonwhite_arr(np.asarray(Image.open(path).convert("RGB")))


def _file_digest(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TestRenderLibrary(unittest.TestCase):
    def setUp(self):
        os.environ.pop("DISPLAY", None)

    def test_iso_writes_png_without_display(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = render(Box(10, 6, 4), view="iso", out=_png_path(tmp), size=(96, 72))

            data = Path(path).read_bytes()
            self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertGreater(len(data), 64)
            image = Image.open(path)
            self.assertEqual(image.size, (96, 72))

    def test_translucent_washes_toward_white(self):
        opaque = Box(12, 8, 5)
        opaque.color = Color(0.9, 0.05, 0.05, 1.0)
        glass = Box(12, 8, 5)
        glass.color = Color(0.9, 0.05, 0.05, 0.25)

        with tempfile.TemporaryDirectory() as tmp:
            opaque_path = render(
                opaque, out=_png_path(tmp, "opaque.png"), size=(96, 72)
            )
            glass_path = render(glass, out=_png_path(tmp, "glass.png"), size=(96, 72))
            opaque_rgb = _mean_nonwhite(str(opaque_path))
            glass_rgb = _mean_nonwhite(str(glass_path))

        self.assertGreater(opaque_rgb[0], opaque_rgb[1] + 40)
        self.assertGreater(opaque_rgb[0], opaque_rgb[1])
        self.assertGreater(glass_rgb[1], opaque_rgb[1])
        self.assertGreater(glass_rgb[2], opaque_rgb[2])
        self.assertGreater(glass_rgb.sum(), opaque_rgb.sum())

    def test_appearances_mapping_overrides_shape_color(self):
        box = Box(8, 8, 8)
        box.color = Color(0.0, 1.0, 0.0)

        with tempfile.TemporaryDirectory() as tmp:
            path = render(
                box,
                out=_png_path(tmp),
                size=(80, 80),
                appearances={box: Color(0.05, 0.1, 0.9, 1.0)},
            )
            rgb = _mean_nonwhite(str(path))

        self.assertGreater(rgb[2], rgb[0] + 30)
        self.assertGreater(rgb[2], rgb[1])

    def test_section_cut_changes_pixel_bytes(self):
        hollow = Box(20, 20, 20) - Box(12, 12, 12)

        with tempfile.TemporaryDirectory() as tmp:
            uncut = render(hollow, out=_png_path(tmp, "uncut.png"), size=(96, 72))
            cut = render(
                hollow,
                out=_png_path(tmp, "cut.png"),
                size=(96, 72),
                section=Plane.YZ,
            )
            self.assertNotEqual(_file_digest(str(uncut)), _file_digest(str(cut)))
            self.assertTrue(Path(uncut).read_bytes().startswith(b"\x89PNG"))
            self.assertTrue(Path(cut).read_bytes().startswith(b"\x89PNG"))

    def test_section_cut_dataclass_keep_bottom(self):
        offset_from_yz = Pos(4, 0, 0) * Box(16, 8, 8)

        with tempfile.TemporaryDirectory() as tmp:
            top = render(
                offset_from_yz,
                out=_png_path(tmp, "top.png"),
                size=(80, 64),
                section=SectionCut(Plane.YZ, keep=Keep.TOP),
            )
            bottom = render(
                offset_from_yz,
                out=_png_path(tmp, "bottom.png"),
                size=(80, 64),
                section=SectionCut(Plane.YZ, keep=Keep.BOTTOM),
            )
            self.assertNotEqual(_file_digest(str(top)), _file_digest(str(bottom)))

    def test_front_view_differs_from_iso(self):
        part = Box(10, 4, 6)

        with tempfile.TemporaryDirectory() as tmp:
            iso = render(
                part, view=ViewPreset.ISO, out=_png_path(tmp, "iso.png"), size=(80, 64)
            )
            front = render(
                part, view="front", out=_png_path(tmp, "front.png"), size=(80, 64)
            )
            self.assertNotEqual(_file_digest(str(iso)), _file_digest(str(front)))

    def test_compound_list_and_jpg_suffix(self):
        a = Box(6, 6, 2)
        a.color = Color("red")
        b = Pos(8, 0, 0) * Box(6, 6, 2)
        b.color = Color("navy", 0.4)
        assy = Compound(children=[a, b])

        with tempfile.TemporaryDirectory() as tmp:
            png = render(assy, out=_png_path(tmp, "assy.png"), size=(96, 72))
            jpg = render([a, b], out=_png_path(tmp, "assy.jpg"), size=(96, 72))
            self.assertTrue(Path(png).read_bytes().startswith(b"\x89PNG"))
            self.assertGreater(Path(jpg).stat().st_size, 32)

    def test_compound_children_keep_per_body_color(self):
        left = Box(6, 6, 2)
        left.color = Color(0.95, 0.05, 0.05, 1.0)
        right = Pos(10, 0, 0) * Box(6, 6, 2)
        right.color = Color(0.05, 0.05, 0.95, 1.0)
        assy = Compound(children=[left, right])

        with tempfile.TemporaryDirectory() as tmp:
            path = render(
                assy,
                view="top",
                out=_png_path(tmp),
                size=(160, 80),
            )
            pixels = np.asarray(Image.open(path).convert("RGB"))

        left_px = pixels[:, :80]
        right_px = pixels[:, 80:]
        left_rgb = _mean_nonwhite_arr(left_px)
        right_rgb = _mean_nonwhite_arr(right_px)
        self.assertGreater(left_rgb[0], left_rgb[2] + 40)
        self.assertGreater(right_rgb[2], right_rgb[0] + 40)

    def test_unknown_view_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                render(Box(1, 1, 1), view="axon", out=_png_path(tmp))

    def test_empty_input_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                render([], out=_png_path(tmp))

    def test_invalid_size_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                render(Box(1, 1, 1), out=_png_path(tmp), size=(0, 10))

    def test_section_keep_both_rejected(self):
        with self.assertRaises(ValueError):
            SectionCut(Plane.XY, keep=Keep.BOTH)

    def test_section_that_misses_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            far = Plane.XY.offset(1000)
            with self.assertRaises(ValueError):
                render(Box(2, 2, 2), out=_png_path(tmp), section=far)

    def test_parallel_look_up_raises(self):
        with self.assertRaises(ValueError):
            _camera_basis(Vector(0, 0, 1), Vector(0, 0, 1))
