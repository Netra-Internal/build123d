"""
build123d probe tests

name: test_probe.py
by:   Netra
date: September 9th 2026

desc: Library-first tests for probe() plus a thin CLI smoke check.
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
from unittest.mock import patch

from build123d import (
    Box,
    Compound,
    Cylinder,
    Pos,
    export_step,
    import_step,
    probe,
)
from build123d.cli import (
    EXIT_EMPTY,
    EXIT_ERROR,
    EXIT_NOT_FOUND,
    EXIT_OK,
    console_main,
    main,
)
from build123d.probe import ProbeResult, _cylinder_radius, _detect_holes


def _assert_vector_almost_equal(test, actual, expected, places=5):
    test.assertAlmostEqual(actual.X, expected[0], places)
    test.assertAlmostEqual(actual.Y, expected[1], places)
    test.assertAlmostEqual(actual.Z, expected[2], places)


def _plate_with_hole():
    """20x20x5 plate, through-hole diameter 4 on +Z through the origin."""
    plate = Box(20, 20, 5) - Cylinder(2, 10)
    plate.label = "plate"
    return plate


def _multi_body_assembly():
    plate = _plate_with_hole()
    scrap = Pos(40, 0, 0) * Box(5, 5, 5)
    scrap.label = "scrap"
    assembly = Compound(children=[plate, scrap])
    assembly.label = "assy"
    return assembly


def _write_step(shape, directory: str, name: str = "part.step") -> str:
    path = os.path.join(directory, name)
    export_step(shape, path)
    return path


class TestProbeLibrary(unittest.TestCase):
    """probe() is the product — inventory, holes, strip."""

    def test_shape_reports_hole_without_step(self):
        plate = _plate_with_hole()
        result = probe(plate)

        self.assertEqual(result.names, ("plate",))
        body = result.bodies[0]
        self.assertEqual(len(body.holes), 1)
        hole = body.holes[0]
        # Evidence: Box(20,20,5) - Cylinder(r=2) through origin along Z.
        self.assertAlmostEqual(hole.diameter, 4.0, places=5)
        _assert_vector_almost_equal(self, hole.center, (0.0, 0.0, 0.0))
        _assert_vector_almost_equal(self, hole.axis, (0.0, 0.0, 1.0))
        self.assertAlmostEqual(body.bbox.min.X, -10.0, places=5)
        self.assertAlmostEqual(body.bbox.max.X, 10.0, places=5)

    def test_multi_body_step_reports_bodies_and_hole(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_step(_multi_body_assembly(), tmp)
            imported = import_step(path)
            result = probe(imported)

        self.assertEqual(set(result.names), {"plate", "scrap"})
        by_name = {body.name: body for body in result.bodies}
        self.assertEqual(len(by_name["plate"].holes), 1)
        self.assertEqual(len(by_name["scrap"].holes), 0)
        hole = by_name["plate"].holes[0]
        self.assertAlmostEqual(hole.diameter, 4.0, places=5)
        _assert_vector_almost_equal(self, hole.center, (0.0, 0.0, 0.0))
        _assert_vector_almost_equal(self, hole.axis, (0.0, 0.0, 1.0))

    def test_probe_path_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_step(_multi_body_assembly(), tmp)
            result = probe(path)

        self.assertEqual(set(result.names), {"plate", "scrap"})
        self.assertTrue(any(body.holes for body in result.bodies))

    def test_strip_by_name_drops_named_body(self):
        result = probe(_multi_body_assembly(), strip=["scrap"])
        self.assertEqual(result.names, ("plate",))
        self.assertEqual(len(result.bodies[0].holes), 1)

        result_str = probe(_multi_body_assembly(), strip="plate")
        self.assertEqual(result_str.names, ("scrap",))
        self.assertEqual(result_str.bodies[0].holes, ())

    def test_strip_is_exact_and_unknown_names_are_ignored(self):
        result = probe(_multi_body_assembly(), strip=["Scrap", "missing"])
        self.assertEqual(set(result.names), {"plate", "scrap"})

    def test_strip_all_returns_empty_result(self):
        result = probe(_multi_body_assembly(), strip=["plate", "scrap"])
        self.assertEqual(result.bodies, ())
        self.assertEqual(result.to_dict(), {"bodies": []})

    def test_outer_cylinder_is_not_a_hole(self):
        pin = Cylinder(3, 8)
        pin.label = "pin"
        result = probe(pin)
        self.assertEqual(result.names, ("pin",))
        self.assertEqual(result.bodies[0].holes, ())

    def test_two_holes_are_reported_separately(self):
        plate = Box(30, 20, 5) - Pos(-8, 0, 0) * Cylinder(2, 10)
        plate = plate - Pos(8, 0, 0) * Cylinder(3, 10)
        result = probe(plate)
        diameters = sorted(hole.diameter for hole in result.bodies[0].holes)
        self.assertEqual(len(diameters), 2)
        self.assertAlmostEqual(diameters[0], 4.0, places=5)
        self.assertAlmostEqual(diameters[1], 6.0, places=5)

    def test_to_dict_is_json_serializable(self):
        result = probe(_plate_with_hole())
        payload = result.to_dict()
        encoded = json.dumps(payload)
        self.assertIn("plate", encoded)
        self.assertIn("diameter", encoded)
        self.assertEqual(payload["bodies"][0]["name"], "plate")
        self.assertIn("min", payload["bodies"][0]["bbox"])
        self.assertIn("max", payload["bodies"][0]["bbox"])

    def test_missing_step_raises(self):
        with self.assertRaises(FileNotFoundError):
            probe("/definitely/not/a/real/file.step")

    def test_radius_fallback_when_face_radius_is_none(self):
        plate = _plate_with_hole()
        face = next(
            f
            for f in plate.faces()
            if f.geom_type.name == "CYLINDER" and f.is_circular_concave
        )
        axis = face.axis_of_rotation
        with patch.object(type(face), "radius", new=property(lambda self: None)):
            radius = _cylinder_radius(face, axis)
        self.assertAlmostEqual(radius, 2.0, places=5)

    def test_convexity_errors_are_skipped(self):
        plate = _plate_with_hole()

        def _raise(self):
            raise ValueError("degenerate")

        with patch.object(
            type(plate.faces()[0]), "is_circular_concave", property(_raise)
        ):
            holes = _detect_holes(plate)
        self.assertEqual(holes, ())


class TestProbeCli(unittest.TestCase):
    """Thin wrapper: non-zero exit + JSON on fail. Library remains the product."""

    def _run(self, argv):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(argv)
        text = buffer.getvalue()
        payload = json.loads(text) if text.strip() else None
        return code, payload, text

    def test_missing_file_exits_2_with_json(self):
        code, payload, _ = self._run(["probe", "/no/such/file.step"])
        self.assertEqual(code, EXIT_NOT_FOUND)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"], "file_not_found")
        self.assertEqual(payload["path"], "/no/such/file.step")

    def test_usage_error_exits_2_with_json(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer), self.assertRaises(SystemExit) as raised:
            main([])
        self.assertEqual(raised.exception.code, EXIT_NOT_FOUND)
        payload = json.loads(buffer.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"], "usage")

    def test_strip_all_exits_3_with_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_step(_multi_body_assembly(), tmp)
            code, payload, _ = self._run(["probe", path, "--strip", "plate", "scrap"])
        self.assertEqual(code, EXIT_EMPTY)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"], "empty_after_strip")
        self.assertEqual(payload["strip"], ["plate", "scrap"])

    def test_success_writes_json_and_exits_0(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_step(_multi_body_assembly(), tmp)
            out = os.path.join(tmp, "report.json")
            code, payload, _ = self._run(["probe", path, "--json", out])
            on_disk = json.loads(Path(out).read_text(encoding="utf-8"))
        self.assertEqual(code, EXIT_OK)
        self.assertTrue(payload["ok"])
        self.assertEqual(
            set(body["name"] for body in payload["bodies"]), {"plate", "scrap"}
        )
        plate = next(body for body in payload["bodies"] if body["name"] == "plate")
        self.assertGreaterEqual(len(plate["holes"]), 1)
        self.assertEqual(on_disk["ok"], True)

    def test_probe_exception_exits_1_with_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_step(_plate_with_hole(), tmp)
            with patch("build123d.cli.probe", side_effect=RuntimeError("boom")):
                code, payload, _ = self._run(["probe", path])
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(payload["error"], "probe_failed")
        self.assertIn("boom", payload["message"])

    def test_console_main_uses_exit_code(self):
        with patch("build123d.cli.main", return_value=2):
            with self.assertRaises(SystemExit) as raised:
                console_main()
        self.assertEqual(raised.exception.code, 2)

    def test_file_disappears_after_stat_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_step(_plate_with_hole(), tmp)
            with patch("build123d.cli.probe", side_effect=FileNotFoundError(path)):
                code, payload, _ = self._run(["probe", path])
        self.assertEqual(code, EXIT_NOT_FOUND)
        self.assertEqual(payload["error"], "file_not_found")

    def test_missing_file_writes_json_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "fail.json")
            code, payload, _ = self._run(["probe", "/no/such/file.step", "--json", out])
            on_disk = json.loads(Path(out).read_text(encoding="utf-8"))
        self.assertEqual(code, EXIT_NOT_FOUND)
        self.assertEqual(on_disk["error"], payload["error"])

    def test_empty_step_exits_3_with_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_step(_plate_with_hole(), tmp)
            with patch("build123d.cli.probe", return_value=ProbeResult(bodies=())):
                code, payload, _ = self._run(["probe", path])
        self.assertEqual(code, EXIT_EMPTY)
        self.assertEqual(payload["error"], "empty")

    def test_nested_assembly_uses_leaf_labels(self):
        plate = _plate_with_hole()
        sub = Compound(children=[plate])
        sub.label = "sub"
        result = probe(Compound(children=[sub]))
        self.assertEqual(result.names, ("plate",))
        self.assertEqual(len(result.bodies[0].holes), 1)


if __name__ == "__main__":
    unittest.main()
