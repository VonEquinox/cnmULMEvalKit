from __future__ import annotations

from pathlib import Path

import pytest

from cnm_t2i import hfwrap, installer, uvwrap


def test_download_marker_skips_subsequent_downloads(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "cnm-t2i-work"

    calls = {"download": 0, "access": 0, "pip": 0, "venv": 0}

    def fake_create_venv(*args, **kwargs):
        calls["venv"] += 1
        return uvwrap.CommandResult(cmd=["uv", "venv"], env=None)

    def fake_pip_install(*args, **kwargs):
        calls["pip"] += 1
        return uvwrap.CommandResult(cmd=["uv", "pip", "install"], env=None)

    def fake_check_access(*args, **kwargs):
        calls["access"] += 1
        return hfwrap.ModelAccess(ok=True, message="ok")

    def fake_download(*args, **kwargs):
        calls["download"] += 1
        return str(kwargs["local_dir"])

    monkeypatch.setattr(uvwrap, "create_venv", fake_create_venv)
    monkeypatch.setattr(uvwrap, "pip_install", fake_pip_install)
    monkeypatch.setattr(hfwrap, "check_model_access", fake_check_access)
    monkeypatch.setattr(hfwrap, "download_snapshot", fake_download)

    req = installer.InstallRequest(
        home=home,
        model_key="sd15",
        hf_repo=None,
        runtime=None,
        family=None,
        revision=None,
        torch_backend="cpu",
        python="3.11",
        managed_python=False,
        hf_token=None,
        hf_endpoint=None,
        env_only=False,
        download_only=False,
        force=False,
        dry_run=False,
    )

    installer.install(req)
    assert calls["download"] == 1
    assert (home / "models" / "sd15" / ".cnm_snapshot_complete.json").is_file()

    installer.install(req)
    # Second install should skip downloading because marker exists.
    assert calls["download"] == 1
    # It should also skip access check on the second run (since no download).
    assert calls["access"] == 1


def test_force_redownload_clears_marker(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "cnm-t2i-work"
    model_dir = home / "models" / "sd15"
    model_dir.mkdir(parents=True)
    (model_dir / ".cnm_snapshot_complete.json").write_text("{}", encoding="utf-8")

    calls = {"download": 0, "access": 0, "pip": 0, "venv": 0}

    monkeypatch.setattr(uvwrap, "create_venv", lambda *a, **k: uvwrap.CommandResult(cmd=["uv", "venv"], env=None))
    monkeypatch.setattr(uvwrap, "pip_install", lambda *a, **k: uvwrap.CommandResult(cmd=["uv", "pip", "install"], env=None))
    monkeypatch.setattr(hfwrap, "check_model_access", lambda *a, **k: hfwrap.ModelAccess(ok=True, message="ok"))

    def fake_download(*args, **kwargs):
        nonlocal calls
        calls["download"] += 1
        return str(kwargs["local_dir"])

    monkeypatch.setattr(hfwrap, "download_snapshot", fake_download)

    req = installer.InstallRequest(
        home=home,
        model_key="sd15",
        hf_repo=None,
        runtime=None,
        family=None,
        revision=None,
        torch_backend="cpu",
        python="3.11",
        managed_python=False,
        hf_token=None,
        hf_endpoint=None,
        env_only=False,
        download_only=False,
        force=True,
        dry_run=False,
    )

    installer.install(req)
    assert calls["download"] == 1
    assert (home / "models" / "sd15" / ".cnm_snapshot_complete.json").is_file()

