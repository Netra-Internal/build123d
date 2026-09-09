"""
build123d mate

name: mate.py
by:   Netra
date: September 9th 2026

desc:
    Place one part onto another by matching a hole or a face. Resolves
    features to Locations, then wraps RigidJoint.connect_to. The fixed
    part stays. The moving part is relocated in place.

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

from dataclasses import dataclass
from itertools import count
from typing import Any, Union

from build123d.geometry import Axis, Location, Plane
from build123d.joints import RigidJoint
from build123d.probe import ProbedBody, ProbedHole, probe
from build123d.topology.composite import Compound
from build123d.topology.shape_core import Shape
from build123d.topology.three_d import Solid
from build123d.topology.two_d import Face

__all__ = [
    "MateResult",
    "mate",
]

JointHost = Union[Solid, Compound]
FeatureRef = Union[int, ProbedHole, Face, Plane, Location]

_MATE_LABELS = count()


@dataclass(frozen=True)
class MateResult:
    """Pose after a RigidJoint mate. ``fixed`` is unchanged. ``moving`` moved."""

    location: Location
    fixed: JointHost
    moving: JointHost
    fixed_joint: RigidJoint
    moving_joint: RigidJoint

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready pose. Joint objects are omitted."""
        position = self.location.position
        orientation = self.location.orientation
        return {
            "location": {
                "position": [position.X, position.Y, position.Z],
                "orientation": [orientation.X, orientation.Y, orientation.Z],
            },
            "fixed": self.fixed.label or "",
            "moving": self.moving.label or "",
        }


def mate(
    fixed: Shape | ProbedBody,
    moving: Shape | ProbedBody,
    *,
    fixed_feature: FeatureRef,
    moving_feature: FeatureRef,
    flip: bool = False,
) -> MateResult:
    """Mate ``moving`` onto ``fixed`` by matching one feature on each.

    An ``int`` is a hole index from :func:`probe`. You may also pass a
    :class:`ProbedHole`, :class:`Face`, :class:`Plane`, or :class:`Location`.

    ``fixed`` stays. ``moving`` is relocated in place so the two feature
    frames coincide. ``flip`` inverts the moving feature's Z before the
    joint is built (face-to-face sit). Missing or unknown features raise
    ``ValueError`` or ``TypeError``.

    Args:
        fixed: Host that stays put.
        moving: Host that is relocated.
        fixed_feature: Feature on ``fixed``.
        moving_feature: Feature on ``moving``.
        flip: Invert the moving feature Z. Defaults to False.

    Returns:
        MateResult: Coincident joint location and both hosts.
    """
    fixed_host = _require_host(fixed, "fixed")
    moving_host = _require_host(moving, "moving")
    fixed_location = _feature_location(fixed, fixed_host, fixed_feature, "fixed")
    moving_location = _feature_location(moving, moving_host, moving_feature, "moving")
    if flip:
        plane = Plane(moving_location)
        moving_location = Location(Plane(origin=plane.origin, z_dir=-plane.z_dir))

    fixed_joint = RigidJoint(
        _joint_label(fixed_host, moving_host), fixed_host, fixed_location
    )
    moving_joint = RigidJoint(
        _joint_label(fixed_host, moving_host), moving_host, moving_location
    )
    fixed_joint.connect_to(moving_joint)
    return MateResult(
        location=fixed_joint.location,
        fixed=fixed_host,
        moving=moving_host,
        fixed_joint=fixed_joint,
        moving_joint=moving_joint,
    )


def _joint_label(*hosts: JointHost) -> str:
    taken = {label for host in hosts for label in host.joints}
    while True:
        label = f"_b123d_mate_{next(_MATE_LABELS)}"
        if label not in taken:
            return label


def _require_host(source: Shape | ProbedBody, role: str) -> JointHost:
    shape = source.shape if isinstance(source, ProbedBody) else source
    if isinstance(shape, (Solid, Compound)):
        return shape
    raise ValueError(f"{role} must be a Solid or Compound, not {type(shape).__name__}")


def _feature_location(
    source: Shape | ProbedBody,
    host: JointHost,
    feature: FeatureRef,
    role: str,
) -> Location:
    if isinstance(feature, Location):
        return feature
    if isinstance(feature, Plane):
        return Location(feature)
    if isinstance(feature, Face):
        return Location(Plane(origin=feature.center(), z_dir=feature.normal_at()))
    if isinstance(feature, ProbedHole):
        return Axis(feature.center, feature.axis).location
    if isinstance(feature, int) and not isinstance(feature, bool):
        return _hole_location(source, host, feature, role)
    raise TypeError(
        f"{role}_feature must be int, ProbedHole, Face, Plane, or Location, "
        f"not {type(feature).__name__}"
    )


def _hole_location(
    source: Shape | ProbedBody,
    host: JointHost,
    index: int,
    role: str,
) -> Location:
    if isinstance(source, ProbedBody):
        holes = source.holes
    else:
        inventory = probe(host)
        if len(inventory.bodies) != 1:
            raise ValueError(
                f"{role} has {len(inventory.bodies)} bodies; "
                "pass a single body or ProbedBody"
            )
        holes = inventory.bodies[0].holes
    if index < 0 or index >= len(holes):
        raise ValueError(
            f"{role}_feature hole index {index} is out of range "
            f"({len(holes)} hole(s))"
        )
    hole = holes[index]
    return Axis(hole.center, hole.axis).location
