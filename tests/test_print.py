# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from math import radians, tan
from pathlib import Path

from build123d import (
    Box,
    BuildPart,
    BuildSketch,
    Compound,
    Plane,
    Polygon,
    Pos,
    export_for_print,
    export_step,
    extrude,
    print_check,
)
from build123d.cli import EXIT_NOT_FOUND, EXIT_OK, main


def _write_step(shape, directory: str, name: str = "part.step") -> str:
    path = os.path.join(directory, name)
    export_step(shape, path)
    return path


def _face_along(part, axis: str, sign: float):
    for face in part.faces():
        normal = face.normal_at()
        component = getattr(normal, axis)
        if abs(component - sign) < 1e-6:
            return face
    raise AssertionError(f"no face with {axis} normal {sign}")


def _overhang_wedge(angle_deg: float):
    """Bed-seated solid with one elevated underside at angle_deg from vertical."""
    run = 6.0
    rise = run / tan(radians(angle_deg))
    z_start = 10.0 - rise
    with BuildPart() as part:
        with BuildSketch(Plane.XZ):
            Polygon((0, 0), (10, 0), (10, z_start), (10 + run, 10), (0, 10))
        extrude(amount=10)
    return part.part


def _cantilever():
    return Box(10, 10, 10) + Pos(10, 0, 5) * Box(20, 10, 4)


class TestPrintCheck(unittest.TestCase):

    def test_plain_box_is_ok(self):
        result = print_check(Box(10, 10, 10))
        self.assertEqual(result.overhangs, ())
        self.assertTrue(result.wall.ok)
        self.assertEqual(result.wall.method, "offset_collapse")
        self.assertIsNotNone(result.wall.measured)
        self.assertAlmostEqual(result.wall.measured, 10.0, places=1)
        self.assertEqual(result.islands, ())
        self.assertTrue(result.ok)

    def test_cantilever_flags_90_degree_underside(self):
        result = print_check(_cantilever())
        self.assertFalse(result.ok)
        self.assertGreaterEqual(len(result.overhangs), 1)
        angles = [record.angle_deg for record in result.overhangs]
        self.assertTrue(any(abs(angle - 90.0) < 1e-4 for angle in angles))
        self.assertAlmostEqual(max(angles), 90.0, places=5)

    def test_thin_plate_fails_wall(self):
        result = print_check(Box(20, 20, 0.4), min_wall=0.8)
        self.assertFalse(result.wall.ok)
        self.assertFalse(result.ok)
        self.assertEqual(result.wall.method, "offset_collapse")
        self.assertEqual(result.wall.min_wall, 0.8)
        self.assertIsNotNone(result.wall.measured)
        self.assertLess(result.wall.measured, 0.8)

    def test_compound_island(self):
        part = Compound(children=[Box(10, 10, 10), Pos(0, 0, 30) * Box(10, 10, 10)])
        result = print_check(part)
        self.assertEqual(len(result.islands), 1)
        self.assertFalse(result.ok)
        island = result.islands[0]
        self.assertEqual(island.name, "solid[1]")
        self.assertAlmostEqual(island.bbox_min.Z, 25.0, places=5)
        self.assertAlmostEqual(island.bbox_max.Z, 35.0, places=5)

    def test_45_degree_does_not_flag_60_does(self):
        wedge45 = _overhang_wedge(45.0)
        flat = print_check(wedge45)
        self.assertEqual(flat.overhangs, ())
        just_under = print_check(wedge45, overhang_deg=44.9)
        self.assertGreaterEqual(len(just_under.overhangs), 1)
        self.assertAlmostEqual(just_under.overhangs[0].angle_deg, 45.0, places=5)

        steep = print_check(_overhang_wedge(60.0))
        self.assertFalse(steep.ok)
        self.assertGreaterEqual(len(steep.overhangs), 1)
        self.assertTrue(
            any(abs(record.angle_deg - 60.0) < 1e-4 for record in steep.overhangs)
        )
        self.assertAlmostEqual(steep.overhangs[0].angle_deg, 60.0, places=5)

    def test_to_dict_is_json_serializable(self):
        result = print_check(_cantilever())
        payload = result.to_dict()
        encoded = json.dumps(payload)
        self.assertIn("overhangs", encoded)
        self.assertIn("offset_collapse", encoded)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["overhang_deg"], 45.0)
        self.assertEqual(payload["min_wall"], 0.8)
        self.assertGreaterEqual(len(payload["overhangs"]), 1)
        self.assertAlmostEqual(payload["overhangs"][0]["angle_deg"], 90.0, places=5)
        self.assertIn("center", payload["overhangs"][0])
        self.assertEqual(len(payload["overhangs"][0]["center"]), 3)
        self.assertEqual(payload["wall"]["method"], "offset_collapse")

    def test_invalid_thresholds_raise(self):
        part = Box(10, 10, 10)
        with self.assertRaises(ValueError):
            print_check(part, overhang_deg=-1)
        with self.assertRaises(ValueError):
            print_check(part, min_wall=-0.1)
        with self.assertRaises(ValueError):
            print_check(part, print_up=(0, 0, 0))
        with self.assertRaises(ValueError):
            print_check(part, overhang_deg=True)

    def test_zero_min_wall_is_ok(self):
        result = print_check(Box(20, 20, 0.4), min_wall=0)
        self.assertTrue(result.wall.ok)


class TestExportForPrint(unittest.TestCase):

    def test_oversize_bed_raises_and_does_not_write(self):
        part = Box(200, 10, 10)
        face_down = part.faces()[0]
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "too_big.stl")
            with self.assertRaises(ValueError) as raised:
                export_for_print(
                    part, face_down=face_down, bed=(180.0, 180.0, 180.0), out=out
                )
            self.assertFalse(os.path.exists(out))
            message = str(raised.exception)
            self.assertIn("180", message)
            self.assertIn("size", message)

    def test_export_rejects_bad_face_and_bed(self):
        part = Box(10, 10, 10)
        face_down = part.faces()[0]
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "bad.stl")
            with self.assertRaises(ValueError):
                export_for_print(part, face_down=part, bed=(180, 180, 180), out=out)
            with self.assertRaises(ValueError):
                export_for_print(
                    part, face_down=face_down, bed=(0, 180, 180), out=out
                )
            with self.assertRaises(ValueError):
                export_for_print(part, face_down=face_down, bed=(180, 180), out=out)
            self.assertFalse(os.path.exists(out))

    def test_successful_export_writes_oriented_stl(self):
        part = Box(10, 20, 40)
        face_down = _face_along(part, "X", -1.0)
        self.assertAlmostEqual(face_down.area, 800.0, places=5)
        original = part.bounding_box().size
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "oriented.stl")
            result = export_for_print(
                part, face_down=face_down, bed=(180.0, 180.0, 180.0), out=out
            )
            self.assertTrue(os.path.isfile(result.path))
            data = Path(out).read_bytes()
        self.assertTrue(result.ok)
        self.assertGreater(len(data), 80)
        self.assertAlmostEqual(result.size[0], 40.0, places=5)
        self.assertAlmostEqual(result.size[1], 20.0, places=5)
        self.assertAlmostEqual(result.size[2], 10.0, places=5)
        self.assertEqual(result.bed, (180.0, 180.0, 180.0))
        self.assertAlmostEqual(original.X, 10.0, places=5)
        self.assertAlmostEqual(original.Y, 20.0, places=5)
        self.assertAlmostEqual(original.Z, 40.0, places=5)
        after = part.bounding_box().size
        self.assertAlmostEqual(after.X, 10.0, places=5)
        self.assertAlmostEqual(after.Y, 20.0, places=5)
        self.assertAlmostEqual(after.Z, 40.0, places=5)
        payload = result.to_dict()
        json.dumps(payload)
        self.assertAlmostEqual(payload["size"][0], 40.0, places=5)
        self.assertAlmostEqual(payload["size"][1], 20.0, places=5)
        self.assertAlmostEqual(payload["size"][2], 10.0, places=5)
        self.assertTrue(payload["ok"])


class TestPrintCheckCli(unittest.TestCase):

    def _run(self, argv):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(argv)
        text = buffer.getvalue()
        payload = json.loads(text) if text.strip() else None
        return code, payload, text

    def test_missing_file_exits_2_with_json(self):
        code, payload, _ = self._run(["print-check", "/no/such/file.step"])
        self.assertEqual(code, EXIT_NOT_FOUND)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"], "file_not_found")
        self.assertEqual(payload["path"], "/no/such/file.step")

    def test_success_writes_nested_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_step(Box(10, 10, 10), tmp)
            out = os.path.join(tmp, "report.json")
            code, payload, _ = self._run(["print-check", path, "--json", out])
            on_disk = json.loads(Path(out).read_text(encoding="utf-8"))
        self.assertEqual(code, EXIT_OK)
        self.assertTrue(payload["ok"])
        report = payload["print_check"]
        self.assertTrue(report["ok"])
        self.assertEqual(report["overhangs"], [])
        self.assertTrue(report["wall"]["ok"])
        self.assertEqual(report["islands"], [])
        self.assertIsNotNone(report["wall"]["measured"])
        self.assertAlmostEqual(report["wall"]["measured"], 10.0, places=1)
        self.assertEqual(on_disk["ok"], True)
        self.assertEqual(on_disk["print_check"]["wall"]["method"], "offset_collapse")

    def test_negative_overhang_is_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_step(Box(10, 10, 10), tmp)
            code, payload, _ = self._run(
                ["print-check", path, "--overhang", "-1"]
            )
        self.assertEqual(code, EXIT_NOT_FOUND)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"], "usage")


if __name__ == "__main__":
    unittest.main()
