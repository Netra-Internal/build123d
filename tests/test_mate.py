"""
build123d mate tests

name: test_mate.py
by:   Netra
date: September 9th 2026

desc: Library-first tests for mate(). Synthetic holes only.
"""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import unittest

from build123d import (
    Axis,
    Box,
    Cylinder,
    Pos,
    mate,
    probe,
)
from build123d.mate import MateResult


def _assert_vector_almost_equal(test, actual, expected, places=5):
    test.assertAlmostEqual(actual.X, expected[0], places)
    test.assertAlmostEqual(actual.Y, expected[1], places)
    test.assertAlmostEqual(actual.Z, expected[2], places)


def _standoff():
    part = Box(8, 8, 12) - Cylinder(1.25, 16)
    part.label = "standoff"
    return part


def _plate():
    part = Pos(30, 10, 0) * (Box(40, 30, 2) - Cylinder(1.4, 6))
    part.label = "plate"
    return part


class TestMateHoles(unittest.TestCase):
    def test_two_holes_produce_expected_relative_pose(self):
        standoff = _standoff()
        plate = _plate()

        result = mate(standoff, plate, fixed_feature=0, moving_feature=0)

        self.assertIsInstance(result, MateResult)
        _assert_vector_almost_equal(self, result.location.position, (0.0, 0.0, 0.0))

        plate_hole = probe(plate).bodies[0].holes[0]
        _assert_vector_almost_equal(self, plate_hole.center, (0.0, 0.0, 0.0))
        _assert_vector_almost_equal(self, plate_hole.axis, (0.0, 0.0, 1.0))

        standoff_box = standoff.bounding_box()
        _assert_vector_almost_equal(self, standoff_box.min, (-4.0, -4.0, -6.0))
        _assert_vector_almost_equal(self, standoff_box.max, (4.0, 4.0, 6.0))

    def test_probed_body_and_hole_inputs(self):
        standoff = _standoff()
        plate = _plate()
        fixed_body = probe(standoff).bodies[0]
        moving_body = probe(plate).bodies[0]

        result = mate(
            fixed_body,
            moving_body,
            fixed_feature=fixed_body.holes[0],
            moving_feature=moving_body.holes[0],
        )

        _assert_vector_almost_equal(self, result.location.position, (0.0, 0.0, 0.0))
        plate_hole = probe(plate).bodies[0].holes[0]
        _assert_vector_almost_equal(self, plate_hole.center, (0.0, 0.0, 0.0))


class TestMateMissingFeatures(unittest.TestCase):
    def test_no_holes_raises(self):
        blank = Box(8, 8, 8)
        plate = _plate()
        with self.assertRaises(ValueError) as ctx:
            mate(blank, plate, fixed_feature=0, moving_feature=0)
        self.assertIn("0 hole", str(ctx.exception))

    def test_out_of_range_hole_raises(self):
        standoff = _standoff()
        plate = _plate()
        with self.assertRaises(ValueError) as ctx:
            mate(standoff, plate, fixed_feature=0, moving_feature=5)
        self.assertIn("5", str(ctx.exception))
        self.assertIn("1 hole", str(ctx.exception))

    def test_non_host_raises(self):
        standoff = _standoff()
        face = standoff.faces()[0]
        with self.assertRaises(ValueError) as ctx:
            mate(face, standoff, fixed_feature=0, moving_feature=0)
        self.assertIn("Solid or Compound", str(ctx.exception))


class TestMateFaces(unittest.TestCase):
    def test_face_mate_sits_parts_face_to_face(self):
        base = Box(10, 10, 4)
        top = Pos(50, 0, 0) * Box(6, 6, 2)
        base_top = base.faces().sort_by(Axis.Z)[-1]
        top_bottom = top.faces().sort_by(Axis.Z)[0]

        result = mate(
            base,
            top,
            fixed_feature=base_top,
            moving_feature=top_bottom,
            flip=True,
        )

        self.assertAlmostEqual(top.bounding_box().min.Z, base.bounding_box().max.Z, 5)
        self.assertAlmostEqual(result.location.position.Z, 2.0, 5)
        self.assertAlmostEqual(base.bounding_box().min.X, -5.0, 5)
