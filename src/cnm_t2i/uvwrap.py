from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


class UvNotFoundError(RuntimeError):
    pass


def require_uv() -> str:
    uv = shutil.which("uv")
    if not uv:
        raise UvNotFoundError("uv not found in PATH. Please install uv first.")
    return uv


def venv_python(venv_dir: Path) -> Path:
    # Linux/macOS virtualenv layout. For Windows, we'd need Scripts/python.exe.
    return venv_dir / "bin" / "python"


def format_cmd(cmd: List[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


@dataclass(frozen=True)
class CommandResult:
    cmd: List[str]
    env: Optional[Dict[str, str]] = None


def run_cmd(
    cmd: List[str],
    *,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[Path] = None,
    dry_run: bool = False,
) -> CommandResult:
    if dry_run:
        return CommandResult(cmd=cmd, env=env)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(cmd, check=True, env=merged_env, cwd=str(cwd) if cwd else None)
    return CommandResult(cmd=cmd, env=env)


def create_venv(
    venv_dir: Path,
    *,
    python: str,
    managed_python: bool = False,
    env: Optional[Dict[str, str]] = None,
    dry_run: bool = False,
) -> CommandResult:
    if not dry_run:
        require_uv()
    if not dry_run and venv_python(venv_dir).is_file():
        return CommandResult(cmd=["(skip)"], env=None)
    cmd = ["uv", "venv", str(venv_dir), "--python", python]
    if managed_python:
        cmd.append("--managed-python")
    return run_cmd(cmd, env=env, dry_run=dry_run)


def pip_install(
    python_exe: Path,
    requirements: Iterable[str],
    *,
    torch_backend: str,
    env: Optional[Dict[str, str]] = None,
    dry_run: bool = False,
) -> CommandResult:
    if not dry_run:
        require_uv()
    cmd = [
        "uv",
        "pip",
        "install",
        "--python",
        str(python_exe),
        "--torch-backend",
        torch_backend,
        *list(requirements),
    ]
    return run_cmd(cmd, env=env, dry_run=dry_run)


def run_python(
    python_exe: Path,
    script_path: Path,
    args: List[str],
    *,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[Path] = None,
    dry_run: bool = False,
) -> CommandResult:
    cmd = [str(python_exe), str(script_path), *args]
    return run_cmd(cmd, env=env, cwd=cwd, dry_run=dry_run)
