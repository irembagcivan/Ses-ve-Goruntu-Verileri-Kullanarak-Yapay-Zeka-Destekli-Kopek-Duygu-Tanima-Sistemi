"""Lightweight YAML configuration loading; no ML imports or side effects."""
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
EMOTION_CLASSES = ("Happy", "Sad", "Angry", "Relaxed")


class ConfigError(ValueError):
    """Configuration syntax or required values are invalid."""


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    return value


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read a complete config; never download weights or inspect dataset files.

    File errors propagate as OSError. Relative paths use the caller's working
    directory; the default path is independent of the working directory.
    """
    source = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    try:
        config = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML: {source}") from exc
    if not isinstance(config, dict):
        raise ConfigError("Configuration must be a mapping")
    project = _section(config, "project")
    if project.get("classes") != list(EMOTION_CLASSES):
        raise ConfigError("project.classes must be [Happy, Sad, Angry, Relaxed]")
    runtime = _section(config, "runtime")
    if runtime.get("device") not in {"auto", "cpu", "mps", "cuda"}:
        raise ConfigError("runtime.device must be auto, cpu, mps or cuda")
    for key, minimum in (("batch_size", 1), ("num_workers", 0)):
        value = runtime.get(key)
        if type(value) is not int or value < minimum:
            raise ConfigError(f"runtime.{key} must be an integer >= {minimum}")
    _section(config, "data")
    _section(config, "preprocessing")
    models = _section(config, "models")
    if _section(models, "detection").get("architecture") != "yolov8":
        raise ConfigError("Detection architecture must be yolov8")
    if _section(models, "segmentation").get("architecture") != "yolov8-seg":
        raise ConfigError("Segmentation architecture must be yolov8-seg")
    if _section(models, "pose").get("species") != "dog":
        raise ConfigError("Pose species must be dog; human pose is unsupported")
    return deepcopy(config)
