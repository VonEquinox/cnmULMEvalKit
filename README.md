# cnm-t2i-installer

Project-local (repo-local) installer/runner for text-to-image (t2i) models using:

- `uv` to create isolated virtualenvs and install runtime deps
- Hugging Face to download model snapshots

All model weights, virtualenvs, HF caches, and outputs are stored under:

`<project_root>/cnm-t2i-work/`

`<project_root>` is detected by walking upwards from the current directory to find `pyproject.toml`.

## Install (on your server)

```bash
uv venv
uv pip install -e ".[dev]"
```

## Usage

List built-in presets:

```bash
cnm-t2i list
```

Interactive setup:

```bash
cnm-t2i setup
```

Non-interactive install:

```bash
cnm-t2i install --model sd15 --torch-backend cu121
```

Dry-run (prints planned actions, does not create envs or download):

```bash
cnm-t2i install --model sd15 --dry-run
```

Smoke inference (after install on server):

```bash
cnm-t2i infer --model sd15 --prompt "a cat" --steps 5
```

## Notes

- Some models may be gated on Hugging Face (you must accept terms and provide a token).
- This repo intentionally keeps the CLI lightweight; heavy deps (torch/diffusers/etc) are installed into `cnm-t2i-work/envs/`.

