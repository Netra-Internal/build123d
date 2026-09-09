"""
build123d fasteners

name: fasteners.py
by:   Netra
date: September 9th 2026

desc:
    Workspace policy for printed-part fasteners. Heat-set inserts are M3
    only (Ø4.0×6.0 on hand). Self-tap pilots are 0.84 × major. Heads
    recess at least 3 mm. Screw length is the next ISO preferred size
    that covers stack + engagement.

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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "InsertSpec",
    "RecessCheck",
    "BomRow",
    "BomResult",
    "self_tap_pilot",
    "insert_spec",
    "check_recess",
    "bom_screws",
]

SELF_TAP_RATIO = 0.84
MIN_RECESS_MM = 3.0
ALLOWED_INSERT_SIZE = "M3"

ISO_PREFERRED_LENGTHS_MM: tuple[float, ...] = (
    4,
    5,
    6,
    8,
    10,
    12,
    16,
    20,
    25,
    30,
    35,
    40,
    45,
    50,
    55,
    60,
)

@dataclass(frozen=True)
class InsertSpec:
    """Insert size, pocket diameter, length, and thread."""

    size: str
    hole_diameter_mm: float
    length_mm: float
    thread: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready insert spec."""
        return {
            "size": self.size,
            "hole_diameter_mm": self.hole_diameter_mm,
            "length_mm": self.length_mm,
            "thread": self.thread,
        }


@dataclass(frozen=True)
class RecessCheck:
    """Head-recess depth against the workspace minimum."""

    depth_mm: float
    minimum_mm: float
    ok: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready recess check."""
        return {
            "depth_mm": self.depth_mm,
            "minimum_mm": self.minimum_mm,
            "ok": self.ok,
        }


@dataclass(frozen=True)
class BomRow:
    """One screw line."""

    qty: int
    size: str
    length_mm: float
    kind: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready BOM row."""
        return {
            "qty": self.qty,
            "size": self.size,
            "length_mm": self.length_mm,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class BomResult:
    """Needed grip plus the ISO preferred screw that covers it."""

    needed_mm: float
    length_mm: float
    rows: tuple[BomRow, ...]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready BOM."""
        return {
            "needed_mm": self.needed_mm,
            "length_mm": self.length_mm,
            "rows": [row.to_dict() for row in self.rows],
        }


_INSERTS = {
    ALLOWED_INSERT_SIZE: InsertSpec(
        size=ALLOWED_INSERT_SIZE,
        hole_diameter_mm=4.0,
        length_mm=6.0,
        thread="M3-0.5",
    )
}

def _normalize_insert_size(size: str) -> str:
    token = size.strip().upper()
    if token in _INSERTS:
        return token
    for spec in _INSERTS.values():
        if token == spec.thread.upper():
            return spec.size
    return token


def self_tap_pilot(major: float) -> float:
    """Return the self-tap pilot diameter. M3 (3.0) is 2.5."""
    if major <= 0:
        raise ValueError(f"major diameter must be positive, got {major}")
    return round(SELF_TAP_RATIO * major, 1)


def insert_spec(size: str) -> InsertSpec:
    """Return the on-hand insert. Non-M3 sizes raise ValueError."""
    key = _normalize_insert_size(size)
    try:
        return _INSERTS[key]
    except KeyError:
        raise ValueError(
            f"workspace policy allows {ALLOWED_INSERT_SIZE} heat-set inserts only; "
            f"got {size}"
        ) from None


def check_recess(depth_mm: float) -> RecessCheck:
    """Return whether head recess meets the 3 mm minimum. Failures do not raise."""
    if depth_mm < 0:
        raise ValueError(f"recess depth must be >= 0, got {depth_mm}")
    return RecessCheck(
        depth_mm=depth_mm,
        minimum_mm=MIN_RECESS_MM,
        ok=depth_mm >= MIN_RECESS_MM,
    )


def _preferred_length(needed_mm: float) -> float:
    for length in ISO_PREFERRED_LENGTHS_MM:
        if length + 1e-9 >= needed_mm:
            return float(length)
    raise ValueError(
        f"needed length {needed_mm} mm exceeds longest ISO preferred "
        f"size {ISO_PREFERRED_LENGTHS_MM[-1]} mm"
    )


def bom_screws(
    stack_mm: Sequence[float],
    engagement_mm: float,
    *,
    qty: int = 1,
    size: str = "M3",
    kind: str = "socket_head_cap",
) -> BomResult:
    """Recommend a screw from stack thicknesses plus engagement."""
    spec = insert_spec(size)
    if not stack_mm:
        raise ValueError("stack_mm must contain at least one thickness")
    if any(layer <= 0 for layer in stack_mm):
        raise ValueError(f"stack thicknesses must be positive, got {tuple(stack_mm)}")
    if engagement_mm <= 0:
        raise ValueError(f"engagement_mm must be positive, got {engagement_mm}")
    if qty < 1:
        raise ValueError(f"qty must be >= 1, got {qty}")

    needed_mm = float(sum(stack_mm) + engagement_mm)
    length_mm = _preferred_length(needed_mm)
    row = BomRow(qty=qty, size=spec.size, length_mm=length_mm, kind=kind)
    return BomResult(needed_mm=needed_mm, length_mm=length_mm, rows=(row,))
