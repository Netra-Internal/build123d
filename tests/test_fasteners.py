"""
build123d fasteners tests

name: test_fasteners.py
by:   Netra
date: September 9th 2026

desc: Synthetic policy tests for M3 pilots, insert reject, BOM length, recess.
"""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

try:
    from build123d import (
        bom_screws,
        check_recess,
        insert_spec,
        self_tap_pilot,
    )
except ModuleNotFoundError:
    _path = Path(__file__).resolve().parents[1] / "src" / "build123d" / "fasteners.py"
    _spec = importlib.util.spec_from_file_location("build123d_fasteners", _path)
    _module = importlib.util.module_from_spec(_spec)
    assert _spec.loader is not None
    sys.modules[_spec.name] = _module
    _spec.loader.exec_module(_module)
    bom_screws = _module.bom_screws
    check_recess = _module.check_recess
    insert_spec = _module.insert_spec
    self_tap_pilot = _module.self_tap_pilot


class TestSelfTapPilot(unittest.TestCase):
    def test_m3_pilot_diameter_is_2_5(self):
        self.assertEqual(self_tap_pilot(3.0), 2.5)

    def test_rejects_non_positive_major(self):
        with self.assertRaises(ValueError):
            self_tap_pilot(0)
        with self.assertRaises(ValueError):
            self_tap_pilot(-3.0)


class TestInsertSpec(unittest.TestCase):
    def test_m3_on_hand_insert(self):
        spec = insert_spec("M3")
        self.assertEqual(spec.size, "M3")
        self.assertEqual(spec.hole_diameter_mm, 4.0)
        self.assertEqual(spec.length_mm, 6.0)
        self.assertEqual(spec.thread, "M3-0.5")
        payload = spec.to_dict()
        self.assertEqual(payload["hole_diameter_mm"], 4.0)
        json.dumps(payload)

    def test_rejects_m4_insert(self):
        with self.assertRaises(ValueError) as ctx:
            insert_spec("M4")
        message = str(ctx.exception)
        self.assertIn("M3", message)
        self.assertIn("M4", message)

    def test_accepts_m3_thread_alias(self):
        self.assertEqual(insert_spec("m3").size, "M3")
        self.assertEqual(insert_spec("M3-0.5").size, "M3")
        self.assertEqual(insert_spec("m3-0.5").size, "M3")


class TestRecess(unittest.TestCase):
    def test_three_mm_passes(self):
        result = check_recess(3.0)
        self.assertTrue(result.ok)
        self.assertEqual(result.minimum_mm, 3.0)
        self.assertEqual(result.depth_mm, 3.0)

    def test_two_mm_fails_without_raising(self):
        result = check_recess(2.0)
        self.assertFalse(result.ok)
        self.assertEqual(result.depth_mm, 2.0)
        self.assertEqual(result.to_dict()["ok"], False)

    def test_just_under_minimum_is_not_ok(self):
        result = check_recess(2.999)
        self.assertFalse(result.ok)
        self.assertEqual(result.depth_mm, 2.999)
        self.assertEqual(result.minimum_mm, 3.0)

    def test_rejects_negative_depth(self):
        with self.assertRaises(ValueError):
            check_recess(-0.1)


class TestBomScrews(unittest.TestCase):
    def test_length_from_stack_geometry(self):
        result = bom_screws((3.0, 2.0), engagement_mm=6.0, qty=4)
        self.assertEqual(result.needed_mm, 11.0)
        self.assertEqual(result.length_mm, 12.0)
        self.assertEqual(len(result.rows), 1)
        row = result.rows[0]
        self.assertEqual(row.qty, 4)
        self.assertEqual(row.size, "M3")
        self.assertEqual(row.length_mm, 12.0)
        payload = result.to_dict()
        self.assertEqual(payload["length_mm"], 12.0)
        self.assertEqual(payload["rows"][0]["qty"], 4)
        json.dumps(payload)

    def test_exact_preferred_length_is_kept(self):
        result = bom_screws((4.0,), engagement_mm=6.0)
        self.assertEqual(result.needed_mm, 10.0)
        self.assertEqual(result.length_mm, 10.0)

    def test_rejects_empty_or_invalid_stack(self):
        with self.assertRaises(ValueError):
            bom_screws((), engagement_mm=6.0)
        with self.assertRaises(ValueError):
            bom_screws((3.0, -1.0), engagement_mm=6.0)
        with self.assertRaises(ValueError):
            bom_screws((3.0,), engagement_mm=0)
        with self.assertRaises(ValueError):
            bom_screws((3.0,), engagement_mm=6.0, qty=0)
        with self.assertRaises(ValueError):
            bom_screws((100.0,), engagement_mm=6.0)


if __name__ == "__main__":
    unittest.main()
