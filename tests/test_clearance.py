# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from build123d import (
    Axis,
    Box,
    Compound,
    Cylinder,
    Pos,
    clearance,
    export_step,
    probe,
)
from build123d.cli import EXIT_NOT_FOUND, EXIT_OK, main


def _write_step(shape, directory: str, name: str = "part.step") -> str:
    path = os.path.join(directory, name)
    export_step(shape, path)
    return path


def _two_boxes_gap_10():
    left = Box(10, 10, 10)
    left.label = "left"
    right = Pos(20, 0, 0) * Box(10, 10, 10)
    right.label = "right"
    return Compound(children=[left, right])


def _three_boxes():
    a = Box(4, 4, 4)
    a.label = "a"
    b = Pos(14, 0, 0) * Box(4, 4, 4)
    b.label = "b"
    c = Pos(0, 20, 0) * Box(4, 4, 4)
    c.label = "c"
    return Compound(children=[a, b, c])


def _overlapping_boxes():
    left = Box(10, 10, 10)
    left.label = "left"
    right = Pos(5, 0, 0) * Box(10, 10, 10)
    right.label = "right"
    return Compound(children=[left, right])


def _wall_and_pin():
    wall = Pos(0, 0, 10) * Box(20, 20, 4)
    wall.label = "wall"
    pin = Cylinder(2, 6)
    pin.label = "pin"
    return Compound(children=[wall, pin])


def _walled_hole_and_pin():
    wall = Pos(0, 0, 10) * (Box(20, 20, 4) - Cylinder(4, 10))
    wall.label = "wall"
    pin = Cylinder(2, 6)
    pin.label = "pin"
    return Compound(children=[wall, pin])


class TestClearanceGaps(unittest.TestCase):

    def test_min_gap_is_real_distance(self):
        result = clearance(_two_boxes_gap_10(), slip=0.2)
        self.assertEqual(len(result.pairs), 1)
        pair = result.pairs[0]
        self.assertEqual(pair.a, "left")
        self.assertEqual(pair.b, "right")
        self.assertAlmostEqual(pair.gap, 10.0, places=5)
        self.assertAlmostEqual(pair.point_a.X, 5.0, places=5)
        self.assertAlmostEqual(pair.point_b.X, 15.0, places=5)
        self.assertTrue(pair.ok)
        self.assertTrue(result.ok)
        self.assertIsNone(result.sweep)

    def test_slip_fail_when_gap_is_short(self):
        result = clearance(_two_boxes_gap_10(), slip=10.1)
        self.assertFalse(result.pairs[0].ok)
        self.assertFalse(result.ok)
        self.assertAlmostEqual(result.pairs[0].gap, 10.0, places=5)

    def test_slip_equal_to_gap_passes(self):
        result = clearance(_two_boxes_gap_10(), slip=10.0)
        self.assertTrue(result.ok)
        self.assertTrue(result.pairs[0].ok)

    def test_overlap_is_zero_gap(self):
        result = clearance(_overlapping_boxes(), slip=0.1)
        self.assertAlmostEqual(result.pairs[0].gap, 0.0, places=5)
        self.assertFalse(result.ok)
        zero_slip = clearance(_overlapping_boxes(), slip=0)
        self.assertTrue(zero_slip.ok)

    def test_three_bodies_make_three_pairs(self):
        result = clearance(_three_boxes(), slip=1.0)
        names = {(pair.a, pair.b) for pair in result.pairs}
        self.assertEqual(names, {("a", "b"), ("a", "c"), ("b", "c")})
        gaps = {(pair.a, pair.b): pair.gap for pair in result.pairs}
        self.assertAlmostEqual(gaps[("a", "b")], 10.0, places=5)
        self.assertAlmostEqual(gaps[("a", "c")], 16.0, places=5)
        self.assertTrue(result.ok)

    def test_accepts_probe_result(self):
        inventory = probe(_two_boxes_gap_10())
        result = clearance(inventory, slip=0.2)
        self.assertAlmostEqual(result.pairs[0].gap, 10.0, places=5)

    def test_accepts_step_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_step(_two_boxes_gap_10(), tmp)
            result = clearance(path, slip=0.2)
        self.assertAlmostEqual(result.pairs[0].gap, 10.0, places=5)

    def test_to_dict_is_json_serializable(self):
        result = clearance(_two_boxes_gap_10(), slip=0.2)
        payload = result.to_dict()
        encoded = json.dumps(payload)
        self.assertIn("left", encoded)
        self.assertIn("right", encoded)
        self.assertEqual(payload["slip"], 0.2)
        self.assertTrue(payload["ok"])
        self.assertIsNone(payload["sweep"])
        self.assertAlmostEqual(payload["pairs"][0]["gap"], 10.0, places=5)

    def test_empty_inventory_is_ok(self):
        result = clearance(Box(1, 1, 1), slip=0.2)
        self.assertEqual(result.pairs, ())
        self.assertTrue(result.ok)

    def test_negative_slip_raises(self):
        with self.assertRaises(ValueError):
            clearance(_two_boxes_gap_10(), slip=-0.1)
        bad: Any = "0.2"
        with self.assertRaises(ValueError):
            clearance(_two_boxes_gap_10(), slip=bad)


class TestClearanceSweep(unittest.TestCase):

    def test_mid_path_hit_fails_when_endpoints_are_clear(self):
        result = clearance(
            _wall_and_pin(),
            slip=0.0,
            moving="pin",
            axis=Axis.Z,
            travel=20,
            steps=5,
        )
        self.assertIsNotNone(result.sweep)
        offsets = [step.offset for step in result.sweep.steps]
        self.assertEqual(offsets, [0.0, 5.0, 10.0, 15.0, 20.0])
        self.assertTrue(result.sweep.steps[0].ok)
        self.assertTrue(result.sweep.steps[-1].ok)
        self.assertEqual(result.sweep.steps[2].colliding, ("wall",))
        self.assertFalse(result.sweep.steps[2].ok)
        self.assertFalse(result.ok)

    def test_clear_hole_sweep_passes(self):
        result = clearance(
            _walled_hole_and_pin(),
            slip=0.0,
            moving="pin",
            axis=Axis.Z,
            travel=20,
            steps=5,
        )
        self.assertTrue(result.sweep.ok)
        self.assertEqual(
            [step.colliding for step in result.sweep.steps],
            [(), (), (), (), ()],
        )
        self.assertTrue(result.ok)

    def test_duplicate_labels_still_detect_mid_hit(self):
        first = Box(10, 10, 10)
        first.label = "box"
        second = Pos(20, 0, 0) * Box(10, 10, 10)
        second.label = "box"
        result = clearance(
            Compound(children=[first, second]),
            slip=0.0,
            moving="box",
            axis=Axis.X,
            travel=20,
            steps=5,
        )
        self.assertFalse(result.ok)
        self.assertTrue(
            any(step.colliding == ("box",) for step in result.sweep.steps)
        )

    def test_moving_index_selects_body(self):
        result = clearance(
            _wall_and_pin(),
            slip=0.0,
            moving=1,
            axis=Axis.Z,
            travel=20,
            steps=3,
        )
        self.assertEqual(result.sweep.moving, "pin")
        self.assertFalse(result.ok)

    def test_partial_sweep_args_raise(self):
        with self.assertRaises(ValueError):
            clearance(_wall_and_pin(), slip=0.0, moving="pin")
        with self.assertRaises(ValueError):
            clearance(_wall_and_pin(), slip=0.0, axis=Axis.Z, travel=20)

    def test_unknown_moving_raises(self):
        with self.assertRaises(ValueError):
            clearance(
                _wall_and_pin(),
                slip=0.0,
                moving="missing",
                axis=Axis.Z,
                travel=20,
            )

    def test_bad_travel_or_steps_raise(self):
        with self.assertRaises(ValueError):
            clearance(
                _wall_and_pin(),
                slip=0.0,
                moving="pin",
                axis=Axis.Z,
                travel=0,
            )
        with self.assertRaises(ValueError):
            clearance(
                _wall_and_pin(),
                slip=0.0,
                moving="pin",
                axis=Axis.Z,
                travel=20,
                steps=1,
            )

    def test_sweep_to_dict_lists_colliding_names(self):
        result = clearance(
            _wall_and_pin(),
            slip=0.0,
            moving="pin",
            axis=Axis.Z,
            travel=20,
            steps=5,
        )
        payload = result.to_dict()
        json.dumps(payload)
        self.assertEqual(payload["sweep"]["moving"], "pin")
        self.assertEqual(payload["sweep"]["axis"], [0.0, 0.0, 1.0])
        self.assertEqual(payload["sweep"]["steps"][2]["colliding"], ["wall"])
        self.assertFalse(payload["ok"])


class TestClearanceCli(unittest.TestCase):

    def _run(self, argv):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(argv)
        text = buffer.getvalue()
        payload = json.loads(text) if text.strip() else None
        return code, payload, text

    def test_missing_file_exits_2_with_json(self):
        code, payload, _ = self._run(
            ["clearance", "/no/such/file.step", "--slip", "0.2"]
        )
        self.assertEqual(code, EXIT_NOT_FOUND)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"], "file_not_found")

    def test_success_writes_nested_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_step(_two_boxes_gap_10(), tmp)
            out = os.path.join(tmp, "report.json")
            code, payload, _ = self._run(
                ["clearance", path, "--slip", "0.2", "--json", out]
            )
            on_disk = json.loads(Path(out).read_text(encoding="utf-8"))
        self.assertEqual(code, EXIT_OK)
        self.assertTrue(payload["ok"])
        report = payload["clearance"]
        self.assertTrue(report["ok"])
        self.assertAlmostEqual(report["pairs"][0]["gap"], 10.0, places=5)
        self.assertEqual(on_disk["ok"], True)

    def test_sweep_flags_detect_mid_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_step(_wall_and_pin(), tmp)
            code, payload, _ = self._run(
                [
                    "clearance",
                    path,
                    "--slip",
                    "0",
                    "--moving",
                    "pin",
                    "--axis",
                    "0",
                    "0",
                    "1",
                    "--travel",
                    "20",
                    "--steps",
                    "5",
                ]
            )
        self.assertEqual(code, EXIT_OK)
        self.assertTrue(payload["ok"])
        self.assertFalse(payload["clearance"]["ok"])
        self.assertEqual(
            payload["clearance"]["sweep"]["steps"][2]["colliding"], ["wall"]
        )


if __name__ == "__main__":
    unittest.main()
