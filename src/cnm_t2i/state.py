from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelInstallRecord(BaseModel):
    key: str
    display_name: Optional[str] = None
    family: Optional[str] = None
    hf_repo: str
    revision: Optional[str] = None
    runtime: str
    torch_backend: str
    python: str
    env_dir: str
    model_dir: str
    installed_at: str = Field(default_factory=utc_now_iso)
    env_ready: bool = True
    downloaded: bool = True


class AppState(BaseModel):
    schema_version: int = 1
    installs: Dict[str, ModelInstallRecord] = Field(default_factory=dict)


def load_state(path: Path) -> AppState:
    if not path.is_file():
        return AppState()
    data = json.loads(path.read_text(encoding="utf-8"))
    return AppState.model_validate(data)


def save_state(path: Path, state: AppState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)

