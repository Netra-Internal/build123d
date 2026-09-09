# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from contextlib import redirect_stdout
from pathlib import Path
from zipfile import ZipFile

from build123d import (
    A1_MINI_BED,
    Box,
    PartDoesNotFitError,
    export_stl,
    inspect_plate_3mf,
    pack_plates,
)
from build123d.cli import EXIT_NOT_FOUND, EXIT_OK, main


def _run_cli(argv):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = main(argv)
    text = buffer.getvalue()
    payload = json.loads(text) if text.strip() else None
    return code, payload, text


class TestPackPlates(unittest.TestCase):

    def test_four_small_boxes_share_plate_one(self):
        parts = [Box(20, 20, 20) for _ in range(4)]
        result = pack_plates(parts)
        self.assertTrue(result.ok)
        self.assertEqual(result.bed, A1_MINI_BED)
        self.assertEqual(result.bed, (180.0, 180.0))
        self.assertEqual(result.edge_margin, 2.0)
        self.assertEqual(result.part_gap, 8.0)
        self.assertEqual(result.usable, (176.0, 176.0))
        self.assertEqual(result.plate_count, 1)
        self.assertIsNone(result.path)
        names = [item.name for item in result.assignments]
        self.assertEqual(names, ["part[0]", "part[1]", "part[2]", "part[3]"])
        self.assertEqual(len(set(names)), 4)
        for item in result.assignments:
            self.assertEqual(item.plate, 1)
            self.assertEqual(item.footprint, (20.0, 20.0))
        self.assertEqual(result.assignments[0].origin_xy, (2.0, 2.0))
        self.assertEqual(result.assignments[1].origin_xy, (30.0, 2.0))
        self.assertEqual(result.assignments[2].origin_xy, (58.0, 2.0))
        self.assertEqual(result.assignments[3].origin_xy, (86.0, 2.0))
        payload = result.to_dict()
        json.dumps(payload)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["plate_count"], 1)
        self.assertEqual(payload["usable"], [176.0, 176.0])
        encoded_names = [item["name"] for item in payload["assignments"]]
        self.assertEqual(encoded_names, names)

    def test_two_plates_when_gap_blocks_a_shared_shelf(self):
        result = pack_plates(
            [Box(20, 20, 20), Box(20, 20, 20)],
            bed=(40.0, 40.0),
            edge_margin=2.0,
            part_gap=8.0,
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.usable, (36.0, 36.0))
        self.assertEqual(result.plate_count, 2)
        names = [item.name for item in result.assignments]
        self.assertEqual(names, ["part[0]", "part[1]"])
        self.assertEqual(result.assignments[0].plate, 1)
        self.assertEqual(result.assignments[1].plate, 2)
        self.assertEqual(result.assignments[0].origin_xy, (2.0, 2.0))
        self.assertEqual(result.assignments[1].origin_xy, (2.0, 2.0))

    def test_later_small_part_fills_earlier_plate(self):
        result = pack_plates(
            [Box(34, 20, 5), Box(34, 20, 5), Box(34, 14, 5)],
            bed=(40.0, 40.0),
            edge_margin=2.0,
            part_gap=2.0,
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.usable, (36.0, 36.0))
        self.assertEqual(result.plate_count, 2)
        self.assertEqual([item.plate for item in result.assignments], [1, 2, 1])
        self.assertEqual(result.assignments[2].origin_xy, (2.0, 24.0))

    def test_oversized_part_raises_and_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "too_big.3mf")
            with self.assertRaises(PartDoesNotFitError) as raised:
                pack_plates(
                    [Box(50, 50, 10)],
                    bed=(40.0, 40.0),
                    edge_margin=2.0,
                    part_gap=8.0,
                    out=out,
                )
            self.assertFalse(os.path.exists(out))
            message = str(raised.exception)
            self.assertIn("part[0]", message)
            self.assertIn("(50.0, 50.0)", message)
            self.assertIn("(36.0, 36.0)", message)

    def test_empty_and_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            pack_plates([])
        with self.assertRaises(ValueError):
            pack_plates([Box(10, 10, 10)], bed=(0, 180))
        with self.assertRaises(ValueError):
            pack_plates([Box(10, 10, 10)], bed=(180,))
        with self.assertRaises(ValueError):
            pack_plates([Box(10, 10, 10)], edge_margin=-1)
        with self.assertRaises(ValueError):
            pack_plates([Box(10, 10, 10)], edge_margin=100)
        with self.assertRaises(ValueError):
            pack_plates([Box(10, 10, 10)], part_gap=True)
        with self.assertRaises(FileNotFoundError):
            pack_plates(["/no/such/file.stl"])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "part.step")
            Path(path).write_text("not-an-stl", encoding="utf-8")
            with self.assertRaises(ValueError):
                pack_plates([path])

    def test_does_not_mutate_caller_shape(self):
        part = Box(20, 20, 20)
        before = part.bounding_box()
        pack_plates([part], bed=(40.0, 40.0), edge_margin=2.0, part_gap=8.0)
        after = part.bounding_box()
        self.assertAlmostEqual(before.min.X, after.min.X, places=5)
        self.assertAlmostEqual(before.min.Y, after.min.Y, places=5)
        self.assertAlmostEqual(before.min.Z, after.min.Z, places=5)
        self.assertAlmostEqual(before.max.X, after.max.X, places=5)


class TestPlate3mfRoundTrip(unittest.TestCase):

    def test_zip_contains_metadata_and_inspect_matches(self):
        parts = [Box(20, 20, 20), Box(20, 20, 20)]
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "plates.3mf")
            result = pack_plates(
                parts,
                bed=(40.0, 40.0),
                edge_margin=2.0,
                part_gap=8.0,
                out=out,
            )
            self.assertEqual(result.path, out)
            self.assertTrue(os.path.isfile(out))
            with ZipFile(out) as archive:
                names = set(archive.namelist())
                netra_raw = archive.read("Metadata/netra_plates.json").decode("utf-8")
                settings_raw = archive.read("Metadata/model_settings.config")
            self.assertIn("[Content_Types].xml", names)
            self.assertIn("_rels/.rels", names)
            self.assertIn("3D/3dmodel.model", names)
            self.assertIn("Metadata/model_settings.config", names)
            self.assertIn("Metadata/netra_plates.json", names)
            netra = json.loads(netra_raw)
            settings = ET.fromstring(settings_raw)
            inspected = inspect_plate_3mf(out)

        self.assertEqual(result.plate_count, 2)
        self.assertEqual(netra["plate_count"], 2)
        self.assertEqual(netra["usable"], [36.0, 36.0])
        written_map = {item["name"]: item["plate"] for item in netra["assignments"]}
        self.assertEqual(written_map, {"part[0]": 1, "part[1]": 2})
        object_ids = {obj.get("id") for obj in settings.findall("object")}
        self.assertEqual(object_ids, {"1", "2"})
        plates = settings.findall("plate")
        self.assertEqual(len(plates), 2)
        on_plate: dict[str, str] = {}
        for plate in plates:
            plater_id = None
            for meta in plate.findall("metadata"):
                if meta.get("key") == "plater_id":
                    plater_id = meta.get("value")
            for instance in plate.findall("model_instance"):
                for meta in instance.findall("metadata"):
                    if meta.get("key") == "object_id":
                        on_plate[meta.get("value")] = plater_id
        self.assertEqual(on_plate, {"1": "1", "2": "2"})
        expected = {
            item["name"]: item["plate"] for item in result.to_dict()["assignments"]
        }
        self.assertEqual(inspected["part_to_plate"], expected)
        self.assertEqual(inspected["part_to_plate"], {"part[0]": 1, "part[1]": 2})
        self.assertEqual(inspected["object_to_plate"], {1: 1, 2: 2})


class TestPackPlatesCli(unittest.TestCase):

    def test_success_writes_nested_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.stl")
            b = os.path.join(tmp, "b.stl")
            export_stl(Box(20, 20, 20), a)
            export_stl(Box(20, 20, 20), b)
            out = os.path.join(tmp, "out.3mf")
            report = os.path.join(tmp, "report.json")
            code, payload, _ = _run_cli(
                ["pack-plates", a, b, "--out", out, "--json", report]
            )
            on_disk = json.loads(Path(report).read_text(encoding="utf-8"))
            self.assertTrue(os.path.isfile(out))
        self.assertEqual(code, EXIT_OK)
        self.assertTrue(payload["ok"])
        report_body = payload["pack_plates"]
        self.assertTrue(report_body["ok"])
        self.assertEqual(report_body["plate_count"], 1)
        self.assertEqual(report_body["usable"], [176.0, 176.0])
        names = [item["name"] for item in report_body["assignments"]]
        self.assertEqual(names, ["a", "b"])
        self.assertEqual(on_disk["ok"], True)
        self.assertEqual(on_disk["pack_plates"]["path"], out)

    def test_missing_file_exits_2_with_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.3mf")
            code, payload, _ = _run_cli(
                ["pack-plates", "/no/such/file.stl", "--out", out]
            )
        self.assertEqual(code, EXIT_NOT_FOUND)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"], "file_not_found")
        self.assertEqual(payload["path"], "/no/such/file.stl")
        self.assertFalse(os.path.exists(out))

    def test_oversized_exits_nonzero_with_ok_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "big.stl")
            export_stl(Box(50, 50, 10), path)
            out = os.path.join(tmp, "out.3mf")
            code, payload, _ = _run_cli(
                [
                    "pack-plates",
                    path,
                    "--out",
                    out,
                    "--bed",
                    "40",
                    "40",
                    "--edge",
                    "2",
                    "--gap",
                    "8",
                ]
            )
        self.assertNotEqual(code, EXIT_OK)
        self.assertEqual(code, EXIT_NOT_FOUND)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"], "does_not_fit")
        self.assertFalse(os.path.exists(out))


if __name__ == "__main__":
    unittest.main()
