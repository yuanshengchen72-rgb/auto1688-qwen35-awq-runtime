from __future__ import annotations

from pathlib import Path

import pytest

from auto1688_art_runtime.runtime import RuntimePreparationError, check_image


ARCHIVE_SIZE = 3_677_681_503


def _valid_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    root.mkdir()
    archive = root / "qwen35-env.tar.zst"
    with archive.open("wb") as handle:
        handle.truncate(ARCHIVE_SIZE)
    restore = root / "restore-qwen35-env.sh"
    restore.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    restore.chmod(0o755)
    return root


def test_check_image_reports_fixed_contract(tmp_path: Path) -> None:
    report = check_image(_valid_root(tmp_path))

    assert report == {
        "status": "image-ready",
        "model_id": "Qwen3.5-35B-A3B-AWQ-4bit",
        "repository": "cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit",
        "python_version": "3.12",
        "vllm_version": "0.23.1rc1.dev1061+g36484e464",
        "model_file_count": 20,
        "model_total_bytes": 24_488_298_779,
        "runtime_archive_bytes": ARCHIVE_SIZE,
    }


def test_check_image_rejects_wrong_archive(tmp_path: Path) -> None:
    root = _valid_root(tmp_path)
    (root / "qwen35-env.tar.zst").write_bytes(b"wrong")

    with pytest.raises(RuntimePreparationError, match="压缩包大小不匹配"):
        check_image(root)
