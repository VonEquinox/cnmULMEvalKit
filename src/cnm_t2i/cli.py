from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import infer as infer_mod
from . import installer, paths, registry


app = typer.Typer(add_completion=False, help="Install and run t2i models into project-local cnm-t2i-work/.")
console = Console()


def _home_option(value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    return Path(value)


@app.command("list")
def list_models() -> None:
    table = Table(title="Built-in model presets")
    table.add_column("key", style="bold")
    table.add_column("family")
    table.add_column("runtime")
    table.add_column("hf_repo")
    table.add_column("gated?")
    table.add_column("notes")
    for m in registry.iter_models():
        table.add_row(
            m.key,
            m.family,
            m.runtime,
            m.hf_repo,
            "yes" if m.maybe_gated else "no",
            m.notes or "",
        )
    console.print(table)


@app.command()
def setup(
    home: Optional[str] = typer.Option(None, "--home", help="Override workdir (default: <project_root>/cnm-t2i-work)"),
) -> None:
    """
    Interactive installer.
    """
    import questionary

    model_choices = [f"{m.key}  ({m.display_name})" for m in registry.iter_models()]
    model_choices.append("custom (provide HF repo + runtime)")
    choice = questionary.select("Select a model preset", choices=model_choices).ask()
    if choice is None:
        raise typer.Exit(code=1)

    model_key: Optional[str] = None
    hf_repo: Optional[str] = None
    runtime: Optional[str] = None
    family: Optional[str] = None

    if choice.startswith("custom"):
        hf_repo = questionary.text("HF repo id (org/name)").ask()
        runtime = questionary.select("Runtime", choices=["diffusers", "janus"]).ask()
        family = questionary.text("Family label (optional)", default="custom").ask()
        if not hf_repo or not runtime:
            raise typer.Exit(code=1)
    else:
        model_key = choice.split()[0]

    use_mirror = bool(
        questionary.confirm(
            "是否使用 Hugging Face 国内镜像 (hf-mirror.com)？", default=False
        ).ask()
    )
    hf_endpoint = "https://hf-mirror.com" if use_mirror else None
    if use_mirror:
        # Allow users to override the mirror URL.
        hf_endpoint = (
            questionary.text(
                "HF Endpoint (可改为你的镜像地址)",
                default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
            ).ask()
            or "https://hf-mirror.com"
        )

    torch_backend = questionary.text("uv torch backend", default="cu121").ask() or "cu121"
    pyver = questionary.text("Python version for venv", default="3.11").ask() or "3.11"
    managed_py = bool(
        questionary.confirm("Use uv managed Python downloads?", default=False).ask()
    )
    do_download = bool(questionary.confirm("Download model snapshot now?", default=True).ask())
    do_env = bool(questionary.confirm("Create/install runtime env now?", default=True).ask())
    dry_run = bool(questionary.confirm("Dry-run (no changes)?", default=False).ask())

    if not do_download and not do_env:
        console.print("[yellow]Nothing selected (env and download both false). Exiting.[/yellow]")
        raise typer.Exit(code=1)

    req = installer.InstallRequest(
        home=_home_option(home),
        model_key=model_key,
        hf_repo=hf_repo,
        runtime=runtime,
        family=family,
        revision=None,
        torch_backend=torch_backend,
        python=pyver,
        managed_python=managed_py,
        hf_token=None,
        hf_endpoint=hf_endpoint,
        env_only=do_env and not do_download,
        download_only=do_download and not do_env,
        force=False,
        dry_run=dry_run,
    )
    installer.install(req)


@app.command()
def install(
    model: Optional[str] = typer.Option(None, "--model", help="Built-in model key (see: cnm-t2i list)"),
    hf_repo: Optional[str] = typer.Option(None, "--hf-repo", help="Custom HF repo id org/name"),
    runtime: Optional[str] = typer.Option(None, "--runtime", help="Runtime for --hf-repo (diffusers/janus)"),
    family: Optional[str] = typer.Option(None, "--family", help="Family label for custom models"),
    revision: Optional[str] = typer.Option(None, "--revision", help="HF revision (branch/tag/sha)"),
    torch_backend: str = typer.Option("cu121", "--torch-backend", help="uv --torch-backend (e.g. cu121/cpu/auto)"),
    python: str = typer.Option("3.11", "--python", help="Python version for uv venv (e.g. 3.11)"),
    managed_python: bool = typer.Option(False, "--managed-python/--no-managed-python", help="Use uv-managed Python downloads"),
    hf_token: Optional[str] = typer.Option(None, "--hf-token", help="HF token (or set HF_TOKEN/HUGGINGFACE_HUB_TOKEN)"),
    hf_endpoint: Optional[str] = typer.Option(None, "--hf-endpoint", help="HF endpoint (or set HF_ENDPOINT)"),
    env_only: bool = typer.Option(False, "--env-only", help="Only create/install env; skip model download"),
    download_only: bool = typer.Option(False, "--download-only", help="Only download model; skip env install"),
    force: bool = typer.Option(False, "--force", help="Force re-download even if model dir exists"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print planned actions; do not execute"),
    home: Optional[str] = typer.Option(None, "--home", help="Override workdir (default: <project_root>/cnm-t2i-work)"),
) -> None:
    req = installer.InstallRequest(
        home=_home_option(home),
        model_key=model,
        hf_repo=hf_repo,
        runtime=runtime,
        family=family,
        revision=revision,
        torch_backend=torch_backend,
        python=python,
        managed_python=managed_python,
        hf_token=hf_token,
        hf_endpoint=hf_endpoint,
        env_only=env_only,
        download_only=download_only,
        force=force,
        dry_run=dry_run,
    )
    installer.install(req)


@app.command()
def infer(
    model: Optional[str] = typer.Option(None, "--model", help="Installed model key (from config.json)"),
    model_path: Optional[str] = typer.Option(None, "--model-path", help="Local model dir (if not using --model)"),
    env_dir: Optional[str] = typer.Option(None, "--env-dir", help="Runtime env dir (if not using --model)"),
    prompt: str = typer.Option(..., "--prompt", help="Text prompt"),
    out: Optional[str] = typer.Option(None, "--out", help="Output image path (default: runs/<ts>_<model>/out.png)"),
    seed: int = typer.Option(0, "--seed"),
    steps: int = typer.Option(30, "--steps"),
    height: int = typer.Option(512, "--height"),
    width: int = typer.Option(512, "--width"),
    guidance: float = typer.Option(4.0, "--guidance"),
    negative: Optional[str] = typer.Option(None, "--negative"),
    true_cfg_scale: float = typer.Option(4.0, "--true-cfg-scale"),
    max_seq_len: int = typer.Option(256, "--max-seq-len"),
    dtype: str = typer.Option("auto", "--dtype", help="auto|fp16|bf16|fp32"),
    device: str = typer.Option("auto", "--device", help="auto|cuda|cpu"),
    gpu: Optional[int] = typer.Option(
        None, "--gpu", help="Select GPU id by setting CUDA_VISIBLE_DEVICES (e.g. 0/1)"
    ),
    cuda_visible_devices: Optional[str] = typer.Option(
        None,
        "--cuda-visible-devices",
        help="Set CUDA_VISIBLE_DEVICES explicitly (e.g. 0 or 0,1). Overrides --gpu.",
    ),
    cfg_weight: float = typer.Option(5.0, "--cfg-weight", help="CFG weight for Janus/JanusFlow"),
    parallel_size: int = typer.Option(1, "--parallel-size", help="Parallel images for Janus (autoreg)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print planned actions; do not execute"),
    home: Optional[str] = typer.Option(None, "--home", help="Override workdir (default: <project_root>/cnm-t2i-work)"),
) -> None:
    if gpu is not None and cuda_visible_devices:
        raise typer.BadParameter("Use either --gpu or --cuda-visible-devices, not both.")
    cuda_vis = cuda_visible_devices if cuda_visible_devices else (str(gpu) if gpu is not None else None)
    req = infer_mod.InferRequest(
        home=_home_option(home),
        model_key=model,
        model_path=Path(model_path) if model_path else None,
        env_dir=Path(env_dir) if env_dir else None,
        prompt=prompt,
        out=Path(out) if out else None,
        seed=seed,
        steps=steps,
        height=height,
        width=width,
        guidance=guidance,
        negative=negative,
        true_cfg_scale=true_cfg_scale,
        max_seq_len=max_seq_len,
        dtype=dtype,
        device=device,
        cuda_visible_devices=cuda_vis,
        cfg_weight=cfg_weight,
        parallel_size=parallel_size,
        dry_run=dry_run,
    )
    out_path = infer_mod.infer(req)
    console.print(f"[green]Output:[/green] {out_path}")


if __name__ == "__main__":
    app()
