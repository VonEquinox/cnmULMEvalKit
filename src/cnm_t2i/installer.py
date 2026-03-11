from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from rich.console import Console

from . import hfwrap, paths, registry, state, uvwrap


console = Console()


@dataclass(frozen=True)
class ResolvedModel:
    key: str
    display_name: str
    family: str
    hf_repo: str
    runtime: str
    maybe_gated: bool


def _require_git_if_needed(runtime: registry.RuntimeSpec) -> None:
    if not runtime.requires_git:
        return
    if not shutil.which("git"):
        raise RuntimeError(
            "git not found in PATH, but this runtime requires installing a Git dependency."
        )


def resolve_model(
    *,
    model_key: Optional[str],
    hf_repo: Optional[str],
    runtime_key: Optional[str],
    family: Optional[str],
) -> ResolvedModel:
    if model_key and hf_repo:
        raise ValueError("Use either --model or --hf-repo, not both")
    if model_key:
        spec = registry.get_model(model_key)
        return ResolvedModel(
            key=spec.key,
            display_name=spec.display_name,
            family=spec.family,
            hf_repo=spec.hf_repo,
            runtime=spec.runtime,
            maybe_gated=spec.maybe_gated,
        )
    if not hf_repo:
        raise ValueError("Must provide either --model or --hf-repo")
    if not runtime_key:
        raise ValueError("When using --hf-repo, you must also provide --runtime")
    safe_key = "hf__" + registry.slugify_repo_id(hf_repo)
    return ResolvedModel(
        key=safe_key,
        display_name=hf_repo,
        family=family or "custom",
        hf_repo=hf_repo,
        runtime=runtime_key,
        maybe_gated=False,
    )


def env_dir_name(runtime_key: str, python: str, torch_backend: str) -> str:
    py_tag = "py" + "".join(c for c in python if c.isdigit())
    tb_tag = torch_backend.replace("/", "_")
    return f"{runtime_key}-{py_tag}-{tb_tag}"


@dataclass(frozen=True)
class InstallRequest:
    home: Optional[Path]
    model_key: Optional[str]
    hf_repo: Optional[str]
    runtime: Optional[str]
    family: Optional[str]
    revision: Optional[str]
    torch_backend: str
    python: str
    managed_python: bool
    hf_token: Optional[str]
    hf_endpoint: Optional[str]
    env_only: bool
    download_only: bool
    force: bool
    dry_run: bool


@dataclass(frozen=True)
class InstallResult:
    layout: paths.HomeLayout
    model: ResolvedModel
    env_dir: Path
    model_dir: Path


def install(req: InstallRequest) -> InstallResult:
    if req.env_only and req.download_only:
        raise ValueError("Cannot use --env-only and --download-only together")
    model = resolve_model(
        model_key=req.model_key,
        hf_repo=req.hf_repo,
        runtime_key=req.runtime,
        family=req.family,
    )
    runtime = registry.get_runtime(model.runtime)
    _require_git_if_needed(runtime)

    home_dir = req.home or paths.default_home_dir()
    layout = paths.HomeLayout.from_home(home_dir)
    env_dir = layout.envs / env_dir_name(runtime.key, req.python, req.torch_backend)
    model_dir = layout.models / model.key

    console.print(f"[bold]Home[/bold]: {layout.home}")
    console.print(f"[bold]Runtime[/bold]: {runtime.key}  -> env: {env_dir}")
    console.print(f"[bold]Model[/bold]: {model.key}  -> dir: {model_dir}")
    if model.maybe_gated:
        console.print(
            "[yellow]Note:[/yellow] This model may be gated on HF; you might need a token."
        )

    if req.dry_run:
        console.print("[cyan](dry-run)[/cyan] Planning actions only; no changes will be made.")

    if not req.dry_run:
        paths.ensure_layout(layout)

    token = hfwrap.resolve_token(req.hf_token)
    endpoint = hfwrap.resolve_endpoint(req.hf_endpoint)

    # Environment setup
    if not req.download_only:
        venv_res = uvwrap.create_venv(
            env_dir,
            python=req.python,
            managed_python=req.managed_python,
            dry_run=req.dry_run,
        )
        py_exe = uvwrap.venv_python(env_dir)
        pip_res = uvwrap.pip_install(
            py_exe,
            runtime.requirements,
            torch_backend=req.torch_backend,
            dry_run=req.dry_run,
        )
        if req.dry_run:
            console.print(f"[cyan]Would run:[/cyan] {uvwrap.format_cmd(venv_res.cmd)}")
            console.print(f"[cyan]Would run:[/cyan] {uvwrap.format_cmd(pip_res.cmd)}")
    else:
        console.print("[cyan]Skipping env setup[/cyan] due to --download-only")

    # Model snapshot download
    if not req.env_only:
        marker = model_dir / ".cnm_snapshot_complete.json"

        if req.force and model_dir.exists():
            console.print("[yellow]Force enabled:[/yellow] clearing existing model directory.")
            if not req.dry_run:
                shutil.rmtree(model_dir)

        if marker.is_file() and not req.force:
            console.print(
                "[green]Model snapshot already marked complete; skipping download.[/green] "
                "Use --force to re-download."
            )
        else:
            access = hfwrap.check_model_access(
                model.hf_repo,
                token=token,
                endpoint=endpoint,
                dry_run=req.dry_run,
            )
            if not access.ok:
                raise RuntimeError(
                    f"Cannot access HF model {model.hf_repo!r}: {access.message}"
                )
            if req.dry_run:
                console.print(
                    "[cyan]Would download:[/cyan] "
                    + json.dumps(
                        {
                            "repo_id": model.hf_repo,
                            "revision": req.revision,
                            "local_dir": str(model_dir),
                            "cache_dir": str(layout.hf_home),
                            "endpoint": endpoint,
                            "token": "(set)" if token else None,
                        },
                        ensure_ascii=False,
                    )
                )
            hfwrap.download_snapshot(
                model.hf_repo,
                local_dir=model_dir,
                cache_dir=layout.hf_home,
                revision=req.revision,
                token=token,
                endpoint=endpoint,
                dry_run=req.dry_run,
            )
            if not req.dry_run:
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(
                    json.dumps(
                        {
                            "repo_id": model.hf_repo,
                            "revision": req.revision,
                            "downloaded_at": state.utc_now_iso(),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
    else:
        console.print("[cyan]Skipping model download[/cyan] due to --env-only")

    # State write
    if not req.dry_run:
        st = state.load_state(layout.state_path)
        st.installs[model.key] = state.ModelInstallRecord(
            key=model.key,
            display_name=model.display_name,
            family=model.family,
            hf_repo=model.hf_repo,
            revision=req.revision,
            runtime=runtime.key,
            torch_backend=req.torch_backend,
            python=req.python,
            env_dir=str(env_dir),
            model_dir=str(model_dir),
            env_ready=not req.download_only,
            downloaded=not req.env_only,
        )
        state.save_state(layout.state_path, st)
    else:
        console.print("[cyan](dry-run)[/cyan] Skipped writing config.json")

    return InstallResult(layout=layout, model=model, env_dir=env_dir, model_dir=model_dir)
