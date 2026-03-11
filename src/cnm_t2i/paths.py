from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_WORKDIR_NAME = "cnm-t2i-work"


def find_project_root(start: Optional[Path] = None) -> Path:
    cur = (start or Path.cwd()).resolve()
    for p in (cur, *cur.parents):
        if (p / "pyproject.toml").is_file():
            return p
    return cur


def default_home_dir(
    start: Optional[Path] = None, workdir_name: str = DEFAULT_WORKDIR_NAME
) -> Path:
    root = find_project_root(start=start)
    return root / workdir_name


@dataclass(frozen=True)
class HomeLayout:
    home: Path
    models: Path
    envs: Path
    hf_home: Path
    runs: Path
    tmp: Path
    state_path: Path

    @staticmethod
    def from_home(home: Path) -> "HomeLayout":
        home = home.resolve()
        return HomeLayout(
            home=home,
            models=home / "models",
            envs=home / "envs",
            hf_home=home / "hf_home",
            runs=home / "runs",
            tmp=home / "tmp",
            state_path=home / "config.json",
        )


def ensure_layout(layout: HomeLayout) -> None:
    layout.home.mkdir(parents=True, exist_ok=True)
    layout.models.mkdir(parents=True, exist_ok=True)
    layout.envs.mkdir(parents=True, exist_ok=True)
    layout.hf_home.mkdir(parents=True, exist_ok=True)
    layout.runs.mkdir(parents=True, exist_ok=True)
    layout.tmp.mkdir(parents=True, exist_ok=True)

