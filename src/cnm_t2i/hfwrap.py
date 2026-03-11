from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import HfHubHTTPError


TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")


def resolve_token(explicit_token: Optional[str]) -> Optional[str]:
    if explicit_token:
        return explicit_token
    for k in TOKEN_ENV_VARS:
        v = os.environ.get(k)
        if v:
            return v
    return None


def resolve_endpoint(explicit_endpoint: Optional[str]) -> Optional[str]:
    if explicit_endpoint:
        return explicit_endpoint
    return os.environ.get("HF_ENDPOINT")


@dataclass(frozen=True)
class ModelAccess:
    ok: bool
    message: str


def check_model_access(
    repo_id: str,
    *,
    token: Optional[str],
    endpoint: Optional[str],
    dry_run: bool = False,
) -> ModelAccess:
    if dry_run:
        return ModelAccess(ok=True, message="(dry-run) skipped access check")
    try:
        api = HfApi(endpoint=endpoint) if endpoint else HfApi()
        _ = api.model_info(repo_id, token=token)
        return ModelAccess(ok=True, message="ok")
    except HfHubHTTPError as e:
        return ModelAccess(ok=False, message=str(e))


def download_snapshot(
    repo_id: str,
    *,
    local_dir: Path,
    cache_dir: Path,
    revision: Optional[str],
    token: Optional[str],
    endpoint: Optional[str],
    dry_run: bool = False,
) -> Optional[str]:
    """
    Download a HF repository snapshot to a local directory.
    Returns the resolved snapshot path from huggingface_hub (or None in dry-run).
    """
    if dry_run:
        return None
    local_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return snapshot_download(
        repo_id=repo_id,
        revision=revision,
        token=token,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        cache_dir=str(cache_dir),
        endpoint=endpoint,
    )

