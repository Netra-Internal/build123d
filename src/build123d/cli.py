"""
build123d CLI

name: cli.py
by:   Netra
date: September 9th 2026

desc:
    Thin agent-facing CLI over the build123d library. The product is the
    Python API; this module only wraps ``probe`` as ``b123d probe``. Do not
    add CRUD / "create-box" style commands here.

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

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NoReturn

from build123d.probe import probe

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NOT_FOUND = 2
EXIT_EMPTY = 3

_EPILOG = """
exit codes:
  0  probe succeeded (JSON with ok=true)
  1  unexpected error
  2  missing file or usage error
  3  no bodies remained (empty STEP or everything stripped)

failures always print a JSON object with ok=false.

examples:
  b123d probe housing.step
  b123d probe housing.step --strip scrap fastener --json out.json
"""


class _JsonArgumentParser(argparse.ArgumentParser):
    """Emit structured JSON on usage errors so agents can parse failures."""

    def error(self, message: str) -> NoReturn:
        _write_json({"ok": False, "error": "usage", "message": message})
        self.exit(EXIT_NOT_FOUND)
        raise SystemExit(EXIT_NOT_FOUND)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``b123d`` CLI. Returns an exit code.

    Usage errors still raise ``SystemExit`` with code 2 (argparse).
    """
    parser = _JsonArgumentParser(
        prog="b123d",
        description=(
            "Thin CLI over the build123d library. "
            "Subcommands wrap library functions; they do not create geometry."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe_parser = subparsers.add_parser(
        "probe",
        help="Inventory STEP bodies, bboxes, and cylindrical holes",
        description=(
            "Load a STEP, optionally strip bodies by exact label, and report "
            "names, bounding boxes, and holes (center, axis, diameter). "
            "Match rule: exact equality on Shape.label after import_step "
            "(import_step replaces space/./() with _)."
        ),
    )
    probe_parser.add_argument("step", help="Path to a STEP file")
    probe_parser.add_argument(
        "--strip",
        action="append",
        nargs="*",
        default=[],
        metavar="NAME",
        help="Exclude bodies whose label equals NAME (exact, repeatable)",
    )
    probe_parser.add_argument(
        "--json",
        dest="json_path",
        metavar="PATH",
        help="Also write the JSON report to PATH",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    strip = [name for group in args.strip for name in group]
    return _cmd_probe(args.step, strip, args.json_path)


def console_main() -> None:
    """Console-script entry point."""
    raise SystemExit(main())


def _cmd_probe(step: str, strip: list[str], json_path: str | None) -> int:
    path = Path(step)
    if not path.is_file():
        payload = {
            "ok": False,
            "error": "file_not_found",
            "message": f"STEP file not found: {step}",
            "path": step,
        }
        _write_json(payload, json_path)
        return EXIT_NOT_FOUND

    try:
        result = probe(path, strip=strip)
    except FileNotFoundError as exc:
        payload = {
            "ok": False,
            "error": "file_not_found",
            "message": str(exc),
            "path": step,
        }
        _write_json(payload, json_path)
        return EXIT_NOT_FOUND
    except Exception as exc:  # pylint: disable=broad-exception-caught
        payload = {
            "ok": False,
            "error": "probe_failed",
            "message": str(exc),
            "path": step,
        }
        _write_json(payload, json_path)
        return EXIT_ERROR

    if not result.bodies:
        error = "empty_after_strip" if strip else "empty"
        payload = {
            "ok": False,
            "error": error,
            "message": (
                "no bodies remained after strip"
                if strip
                else "STEP contained no solid bodies"
            ),
            "path": step,
            "strip": list(strip),
        }
        _write_json(payload, json_path)
        return EXIT_EMPTY

    payload = {"ok": True, "path": str(path), **result.to_dict()}
    _write_json(payload, json_path)
    return EXIT_OK


def _write_json(payload: dict[str, Any], json_path: str | None = None) -> None:
    text = json.dumps(payload, indent=2)
    print(text)
    if json_path is not None:
        Path(json_path).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    console_main()
