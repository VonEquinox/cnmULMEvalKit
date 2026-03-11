from __future__ import annotations

import time
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from rich.console import Console

from . import paths, runner_templates, state, uvwrap


console = Console()


def _default_out(layout: paths.HomeLayout, model_key: str) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return layout.runs / f"{ts}_{model_key}" / "out.png"


def _write_tmp_script(layout: paths.HomeLayout, name: str, content: str) -> Path:
    layout.tmp.mkdir(parents=True, exist_ok=True)
    p = layout.tmp / name
    p.write_text(content, encoding="utf-8")
    return p


def _script_path_for(layout: paths.HomeLayout, name: str, content: str, *, dry_run: bool) -> Path:
    if dry_run:
        # No filesystem side effects in dry-run. The path is only for display.
        return layout.tmp / name
    return _write_tmp_script(layout, name, content)


@dataclass(frozen=True)
class InferRequest:
    home: Optional[Path]
    model_key: Optional[str]
    model_path: Optional[Path]
    env_dir: Optional[Path]
    prompt: str
    out: Optional[Path]
    seed: int
    steps: int
    height: int
    width: int
    guidance: float
    negative: Optional[str]
    true_cfg_scale: float
    max_seq_len: int
    dtype: str
    device: str
    cuda_visible_devices: Optional[str]
    cfg_weight: float
    parallel_size: int
    dry_run: bool


def infer(req: InferRequest) -> Path:
    layout = paths.HomeLayout.from_home(req.home or paths.default_home_dir())
    if not req.dry_run:
        paths.ensure_layout(layout)

    record: Optional[state.ModelInstallRecord] = None
    model_key = req.model_key
    model_dir: Optional[Path] = req.model_path
    env_dir: Optional[Path] = req.env_dir
    family: str = "custom"
    runtime: Optional[str] = None

    if model_key:
        st = state.load_state(layout.state_path)
        record = st.installs.get(model_key)
        if not record:
            raise RuntimeError(
                f"Model {model_key!r} not found in {layout.state_path}. Run install first."
            )
        model_dir = Path(record.model_dir)
        env_dir = Path(record.env_dir)
        family = record.family or "custom"
        runtime = record.runtime

    if not model_dir or not env_dir:
        raise ValueError("Must provide --model (installed) or --model-path + --env-dir")

    py_exe = uvwrap.venv_python(env_dir)
    if not req.dry_run and not py_exe.is_file():
        raise RuntimeError(f"Python not found in env: {py_exe}")

    out_path = req.out or _default_out(layout, model_key or "custom")

    env: Dict[str, str] = {
        "HF_HOME": str(layout.hf_home),
        "HF_HUB_CACHE": str(layout.hf_home / "hub"),
        "TRANSFORMERS_CACHE": str(layout.hf_home / "transformers"),
        "TOKENIZERS_PARALLELISM": "false",
    }
    if record:
        # Do not override user-exported env vars unless explicitly requested.
        if getattr(record, "hf_endpoint", None) and "HF_ENDPOINT" not in os.environ:
            env["HF_ENDPOINT"] = str(record.hf_endpoint)
        if getattr(record, "proxy", None):
            proxy = str(record.proxy)
            for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
                if k not in os.environ:
                    env[k] = proxy
    if req.cuda_visible_devices:
        # Restrict visible GPUs for the subprocess. This is the simplest/most robust way
        # to select a specific GPU without changing downstream libraries.
        env["CUDA_VISIBLE_DEVICES"] = req.cuda_visible_devices

    # Choose runner based on runtime/family.
    if runtime == "janus" or family.startswith("janus"):
        script = runner_templates.janus_runner_script()
        script_path = _script_path_for(layout, "janus_runner.py", script, dry_run=req.dry_run)
        if family == "janus-pro":
            runner_family = "janus-pro"
        elif family == "janusflow":
            runner_family = "janusflow"
        else:
            runner_family = "janus"
        args = [
            "--model-path",
            str(model_dir),
            "--family",
            runner_family,
            "--prompt",
            req.prompt,
            "--out",
            str(out_path),
            "--seed",
            str(req.seed),
            "--device",
            req.device,
            "--dtype",
            req.dtype,
            "--cfg-weight",
            str(req.cfg_weight),
            "--parallel-size",
            str(req.parallel_size),
            "--steps",
            str(req.steps),
        ]
        # Temperature is only used for autoregressive Janus/Janus-Pro.
        if runner_family != "janusflow":
            args.extend(["--temperature", "1.0"])
    else:
        script = runner_templates.diffusers_runner_script()
        script_path = _script_path_for(layout, "diffusers_runner.py", script, dry_run=req.dry_run)
        if family == "flux":
            runner_family = "flux"
        elif family == "qwen-image":
            runner_family = "qwen-image"
        elif family == "stable-diffusion":
            runner_family = "stable-diffusion"
        else:
            runner_family = "auto"
        args = [
            "--model-path",
            str(model_dir),
            "--family",
            runner_family,
            "--prompt",
            req.prompt,
            "--negative",
            req.negative or "",
            "--out",
            str(out_path),
            "--seed",
            str(req.seed),
            "--steps",
            str(req.steps),
            "--height",
            str(req.height),
            "--width",
            str(req.width),
            "--guidance",
            str(req.guidance),
            "--dtype",
            req.dtype,
            "--device",
            req.device,
            "--true-cfg-scale",
            str(req.true_cfg_scale),
            "--max-seq-len",
            str(req.max_seq_len),
        ]

    console.print(f"[bold]Running[/bold] {script_path} in env {env_dir}")
    uvwrap.run_python(py_exe, script_path, args, env=env, dry_run=req.dry_run)
    return out_path
