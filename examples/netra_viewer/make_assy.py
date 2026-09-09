"""
build123d netra viewer example

name: make_assy.py
by:   Netra
date: September 9th 2026

desc: Plate plus translucent cylinder. Writes assy.html and assy.png.
"""

from __future__ import annotations

from pathlib import Path

from build123d import Box, Color, Cylinder, Pos, render, show

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    printed = Box(40, 20, 6)
    printed.color = Color("orange")
    imported = Pos(0, 0, 0) * Cylinder(4, 24)
    imported.color = Color("steelblue", 0.35)
    shapes = [printed, imported]
    show(shapes, out=HERE / "assy.html")
    render(shapes, view="iso", out=HERE / "assy.png")
