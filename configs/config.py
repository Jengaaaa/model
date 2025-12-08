import os
from typing import Any, Dict

import yaml


def _default_config_path() -> str:
    """
    기본 config 경로:
    - 환경변수 JENGA_CONFIG 가 있으면 그 값을 사용
    - 아니면 현재 파일(config.py) 기준 default.yaml
    """
    env_path = os.environ.get("JENGA_CONFIG")
    if env_path:
        return env_path

    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "default.yaml")


def load_config(path: str | None = None) -> Dict[str, Any]:
    if path is None:
        path = _default_config_path()

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg


# 전역에서 바로 쓸 수 있는 기본 설정
CONFIG: Dict[str, Any] = load_config()


