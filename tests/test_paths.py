from __future__ import annotations

from pathlib import Path

from cnm_t2i import paths


def test_find_project_root_walks_up(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "repo"
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    deep = root / "a" / "b" / "c"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)
    assert paths.find_project_root() == root


def test_default_home_dir_is_under_project_root(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "repo"
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    deep = root / "x" / "y"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)
    assert paths.default_home_dir() == root / paths.DEFAULT_WORKDIR_NAME
