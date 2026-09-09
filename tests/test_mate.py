"""
build123d mate tests

name: test_mate.py
by:   Netra
date: September 9th 2026

desc: Library-first tests for mate() — hole, face, and ProbedBody placement.
"""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import json
import unittest

from build123d import (
    Axis,
    Box,
    Compound,
    Cylinder,
    Location,
    Pos,
    RigidJoint,
    mate,
    probe,
)


def _assert_vector_almost_equal(test, actual, expected, places=5):
    test.assertAlmostEqual(actual.X, expected[0], places)
    test.assertAlmostEqual(actual.Y, expected[1], places)
    test.assertAlmostEqual(actual.Z, expected[2], places)


def _standoff():
    """8x8x12 block, through-hole diameter 2.5 on +Z through the origin."""
    return Box(8, 8, 12) - Cylinder(1.25, 16)


def _plate():
    """40x30x2 plate offset to (30, 10, 0), through-hole diameter 2.8."""
    return Pos(30, 10, 0) * (Box(40, 30, 2) - Cylinder(1.4, 6))


class TestMateHoles(unittest.TestCase):
    """Hole index mate relocates the plate onto the standoff."""

    def test_plate_hole_lands_on_standoff_hole(self):
        standoff = _standoff()
        plate = _plate()

        result = mate(standoff, plate, fixed_feature=0, moving_feature=0)

        plate_hole = probe(plate).bodies[0].holes[0]
        _assert_vector_almost_equal(self, plate_hole.center, (0.0, 0.0, 0.0))
        _assert_vector_almost_equal(self, plate_hole.axis, (0.0, 0.0, 1.0))
        _assert_vector_almost_equal(self, result.location.position, (0.0, 0.0, 0.0))
        _assert_vector_almost_equal(self, plate.location.position, (0.0, 0.0, 0.0))

        bbox = standoff.bounding_box()
        self.assertAlmostEqual(bbox.min.X, -4.0, places=5)
        self.assertAlmostEqual(bbox.min.Y, -4.0, places=5)
        self.assertAlmostEqual(bbox.min.Z, -6.0, places=5)
        self.assertAlmostEqual(bbox.max.X, 4.0, places=5)
        self.assertAlmostEqual(bbox.max.Y, 4.0, places=5)
        self.assertAlmostEqual(bbox.max.Z, 6.0, places=5)

        payload = result.to_dict()
        encoded = json.dumps(payload)
        self.assertIn("position", encoded)
        self.assertNotIn("fixed_joint", payload)
        self.assertNotIn("moving_joint", payload)
        self.assertAlmostEqual(payload["location"]["position"][0], 0.0, places=5)
        self.assertAlmostEqual(payload["location"]["position"][1], 0.0, places=5)
        self.assertAlmostEqual(payload["location"]["position"][2], 0.0, places=5)
        self.assertEqual(len(payload["location"]["orientation"]), 3)

    def test_probed_body_and_probed_hole_inputs(self):
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

        plate_hole = probe(moving_body.shape).bodies[0].holes[0]
        _assert_vector_almost_equal(self, plate_hole.center, (0.0, 0.0, 0.0))
        _assert_vector_almost_equal(self, plate_hole.axis, (0.0, 0.0, 1.0))
        _assert_vector_almost_equal(self, result.location.position, (0.0, 0.0, 0.0))

        again_fixed = probe(_standoff()).bodies[0]
        again_moving = probe(_plate()).bodies[0]
        indexed = mate(
            again_fixed, again_moving, fixed_feature=0, moving_feature=0
        )
        _assert_vector_almost_equal(self, indexed.location.position, (0.0, 0.0, 0.0))
        _assert_vector_almost_equal(
            self,
            probe(again_moving.shape).bodies[0].holes[0].center,
            (0.0, 0.0, 0.0),
        )


class TestMateBadFeatures(unittest.TestCase):
    """Missing and illegal features raise; they do not return ok=false."""

    def test_box_with_no_holes_raises(self):
        with self.assertRaisesRegex(
            ValueError, r"fixed_feature hole index 0 is out of range \(0 hole"
        ):
            mate(Box(8, 8, 8), _plate(), fixed_feature=0, moving_feature=0)

    def test_hole_index_out_of_range_raises(self):
        with self.assertRaisesRegex(
            ValueError, r"moving_feature hole index 5 is out of range \(1 hole"
        ):
            mate(_standoff(), _plate(), fixed_feature=0, moving_feature=5)

    def test_non_solid_host_raises(self):
        face = Box(8, 8, 8).faces()[0]
        with self.assertRaisesRegex(ValueError, r"fixed must be a Solid or Compound"):
            mate(face, _standoff(), fixed_feature=face, moving_feature=0)

    def test_int_feature_on_multi_body_raises(self):
        left = _standoff()
        left.label = "left"
        right = Pos(20, 0, 0) * Box(4, 4, 4)
        right.label = "right"
        assembly = Compound(children=[left, right])
        with self.assertRaisesRegex(ValueError, r"single body or ProbedBody"):
            mate(assembly, _plate(), fixed_feature=0, moving_feature=0)

    def test_unknown_feature_type_raises(self):
        with self.assertRaisesRegex(TypeError, r"fixed_feature must be"):
            mate(_standoff(), _plate(), fixed_feature="0", moving_feature=0)


class TestMateFaces(unittest.TestCase):
    """Face mates use center + normal_at, not the planar-face corner origin."""

    def test_face_to_face_with_flip_stacks_boxes(self):
        block_a = Box(10, 10, 10)
        block_b = Pos(25, 4, 0) * Box(10, 10, 10)
        top_a = block_a.faces().sort_by(Axis.Z)[-1]
        bottom_b = block_b.faces().sort_by(Axis.Z)[0]

        self.assertAlmostEqual(top_a.center().X, 0.0, places=5)
        self.assertAlmostEqual(top_a.center().Y, 0.0, places=5)
        self.assertAlmostEqual(top_a.center().Z, 5.0, places=5)

        mate(
            block_a,
            block_b,
            fixed_feature=top_a,
            moving_feature=bottom_b,
            flip=True,
        )

        a_bbox = block_a.bounding_box()
        b_bbox = block_b.bounding_box()
        self.assertAlmostEqual(b_bbox.min.Z, a_bbox.max.Z, places=5)
        self.assertAlmostEqual(a_bbox.max.Z, 5.0, places=5)
        self.assertAlmostEqual(b_bbox.min.X, -5.0, places=5)
        self.assertAlmostEqual(b_bbox.min.Y, -5.0, places=5)
        self.assertAlmostEqual(b_bbox.max.X, 5.0, places=5)
        self.assertAlmostEqual(b_bbox.max.Y, 5.0, places=5)
        self.assertAlmostEqual(b_bbox.max.Z, 15.0, places=5)


class TestMateJoints(unittest.TestCase):

    def test_does_not_overwrite_existing_joint_label(self):
        standoff = _standoff()
        existing = RigidJoint("_b123d_mate_0", standoff, Location())
        result = mate(standoff, _plate(), fixed_feature=0, moving_feature=0)
        self.assertIs(standoff.joints["_b123d_mate_0"], existing)
        self.assertNotEqual(result.fixed_joint.label, "_b123d_mate_0")
        self.assertTrue(result.fixed_joint.label.startswith("_b123d_mate_"))
        self.assertNotEqual(result.fixed_joint.label, result.moving_joint.label)


if __name__ == "__main__":
    unittest.main()
