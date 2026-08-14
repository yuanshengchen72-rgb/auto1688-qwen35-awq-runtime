from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any


MODEL_DIRECTORY_NAME = "Qwen3.5-35B-A3B-AWQ-4bit"
PUBLIC_PATH_RE = re.compile(
    r"/[.]autodl/[0-9a-f]{2}/[0-9a-f]{2}/[0-9a-f]{2}/[0-9a-f]{32}"
)


class RuntimePreparationError(RuntimeError):
    pass


def _contract() -> dict[str, Any]:
    path = Path(__file__).with_name("runtime_contract.json")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimePreparationError("运行契约缺失或损坏。") from exc
    files = value.get("files")
    if not isinstance(files, dict) or len(files) != 20:
        raise RuntimePreparationError("运行契约中的模型文件集合不正确。")
    return value


def check_image(root: Path = Path("/root")) -> dict[str, Any]:
    contract = _contract()
    archive = root / contract["runtime_archive"]["name"]
    restore = root / "restore-qwen35-env.sh"
    if not archive.is_file():
        raise RuntimePreparationError(f"镜像缺少运行环境压缩包：{archive}")
    expected_archive_size = int(contract["runtime_archive"]["size"])
    if archive.stat().st_size != expected_archive_size:
        raise RuntimePreparationError(
            f"运行环境压缩包大小不匹配：{archive.stat().st_size} != {expected_archive_size}"
        )
    if not restore.is_file() or not os.access(restore, os.X_OK):
        raise RuntimePreparationError(f"镜像缺少可执行恢复脚本：{restore}")
    return {
        "status": "image-ready",
        "model_id": contract["model_id"],
        "repository": contract["repository"],
        "python_version": contract["python_version"],
        "vllm_version": contract["vllm_version"],
        "model_file_count": len(contract["files"]),
        "model_total_bytes": sum(int(size) for size in contract["files"].values()),
        "runtime_archive_bytes": expected_archive_size,
    }


def _environment_versions(python: Path) -> tuple[str, str] | None:
    if not python.is_file():
        return None
    completed = subprocess.run(
        [
            str(python),
            "-c",
            "import sys, vllm; "
            "print(f'{sys.version_info.major}.{sys.version_info.minor}'); "
            "print(vllm.__version__)",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        return None
    lines = completed.stdout.strip().splitlines()
    return (lines[0], lines[1]) if len(lines) == 2 else None


def _restore_environment(root: Path, contract: dict[str, Any]) -> Path:
    environment = root / "autodl-tmp" / "qwen35-env"
    python = environment / "bin" / "python"
    expected = (contract["python_version"], contract["vllm_version"])
    if _environment_versions(python) == expected:
        return python
    if environment.exists():
        raise RuntimePreparationError(
            f"同名运行环境存在但版本不正确，拒绝覆盖：{environment}"
        )
    restore = root / "restore-qwen35-env.sh"
    completed = subprocess.run(
        [str(restore)],
        check=False,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if completed.returncode != 0:
        raise RuntimePreparationError(
            "固定运行环境恢复失败：" + (completed.stderr.strip() or completed.stdout.strip())
        )
    if _environment_versions(python) != expected:
        raise RuntimePreparationError("恢复后的 Python/vLLM 版本不符合固定运行契约。")
    return python


def _clean_staging(staging: Path) -> None:
    if staging.is_symlink() or (staging.exists() and not staging.is_dir()):
        staging.unlink()
    elif staging.is_dir():
        shutil.rmtree(staging)


def _verify_model(model: Path, contract: dict[str, Any]) -> None:
    expected_files = contract["files"]
    actual_names = {item.name for item in model.iterdir()}
    if actual_names != set(expected_files):
        raise RuntimePreparationError("模型文件集合不符合固定运行契约。")
    for name, expected_size in expected_files.items():
        path = model / name
        if not path.is_file() or path.stat().st_size != int(expected_size):
            raise RuntimePreparationError(f"模型文件大小不匹配：{name}")
    try:
        config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimePreparationError("模型 config.json 无法解析。") from exc
    quantization = config.get("quantization_config") or {}
    groups = quantization.get("config_groups") or {}
    bits = {
        (group.get("weights") or {}).get("num_bits")
        for group in groups.values()
        if isinstance(group, dict)
    }
    bits.discard(None)
    if (
        config.get("model_type") != "qwen3_5_moe"
        or quantization.get("quant_method") != "compressed-tensors"
        or quantization.get("quantization_status") != "compressed"
        or bits != {4}
    ):
        raise RuntimePreparationError("模型不是固定的 Qwen3.5 AWQ 4-bit 检查点。")


def _link_public_model(
    root: Path,
    filesystem_root: Path,
    contract: dict[str, Any],
) -> Path:
    model = root / "autodl-tmp" / "models" / MODEL_DIRECTORY_NAME
    if model.exists():
        _verify_model(model, contract)
        return model
    manifest_path = root / "auto1688-art-model-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimePreparationError(f"公共模型清单缺失或损坏：{manifest_path}") from exc
    if (
        manifest.get("repository") != contract["repository"]
        or manifest.get("model_id") != contract["model_id"]
    ):
        raise RuntimePreparationError("公共模型清单指向了错误的模型。")
    records = manifest.get("files") or []
    by_name = {
        item.get("name"): item for item in records if isinstance(item, dict)
    }
    expected_files = contract["files"]
    if set(by_name) != set(expected_files):
        raise RuntimePreparationError("公共模型清单文件集合不完整。")
    staging = Path(str(model) + ".art-linking")
    _clean_staging(staging)
    staging.mkdir(parents=True)
    try:
        for name, expected_size in expected_files.items():
            record = by_name[name]
            source_text = str(record.get("autodl_path") or "")
            if not PUBLIC_PATH_RE.fullmatch(source_text):
                raise RuntimePreparationError(f"公共模型路径格式不正确：{name}")
            if int(record.get("size", -1)) != int(expected_size):
                raise RuntimePreparationError(f"公共模型清单大小不正确：{name}")
            source = filesystem_root / source_text.lstrip("/")
            if not source.is_file() or source.stat().st_size != int(expected_size):
                raise RuntimePreparationError(f"公共模型文件不可用：{name}")
            os.symlink(source_text, staging / name)
        staging.rename(model)
    except Exception:
        _clean_staging(staging)
        raise
    _verify_model(model, contract)
    return model


def prepare_runtime(
    root: Path = Path("/root"),
    filesystem_root: Path = Path("/"),
) -> dict[str, Any]:
    image = check_image(root)
    contract = _contract()
    python = _restore_environment(root, contract)
    model = _link_public_model(root, filesystem_root, contract)
    return {
        **image,
        "status": "runtime-ready",
        "python": str(python),
        "model_directory": str(model),
    }
