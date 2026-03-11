from __future__ import annotations

from pathlib import Path

from cnm_t2i import installer


def test_install_dry_run_has_no_filesystem_side_effects(tmp_path: Path) -> None:
    home = tmp_path / "cnm-t2i-work"
    assert not home.exists()

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
        proxy=None,
        env_only=False,
        download_only=False,
        force=False,
        dry_run=True,
    )
    installer.install(req)

    # dry-run should not create the home directory nor write config.
    assert not home.exists()
    assert not (home / "config.json").exists()
