from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from auto1688_art_runtime.runtime import (
    RuntimePreparationError,
    check_image,
    prepare_runtime,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify or prepare the Auto1688 Art runtime")
    parser.add_argument("action", choices=("check-image", "prepare"))
    parser.add_argument("--root", type=Path, default=Path("/root"), help=argparse.SUPPRESS)
    parser.add_argument(
        "--filesystem-root",
        type=Path,
        default=Path("/"),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    try:
        if args.action == "check-image":
            result = check_image(args.root)
        else:
            result = prepare_runtime(args.root, args.filesystem_root)
    except RuntimePreparationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
