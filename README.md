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

注意：如果你没有 `source .venv/bin/activate`，那么 `cnm-t2i` 不会在当前 shell 的 `PATH` 里。
你可以任选一种方式运行：

```bash
source .venv/bin/activate
cnm-t2i list

# 或者不激活 venv：
.venv/bin/cnm-t2i list
# 或者：
uv run cnm-t2i list
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

Hugging Face 国内镜像（可选）：

```bash
cnm-t2i install --model sd15 --hf-endpoint https://hf-mirror.com
# 或者：export HF_ENDPOINT=https://hf-mirror.com
```

代理（可选，比如本机 `clash`）：

```bash
cnm-t2i install --model sd15 --proxy http://127.0.0.1:7897
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

指定 GPU（可选）：

```bash
cnm-t2i infer --model sd15 --prompt "a cat" --gpu 1
# 等价：
cnm-t2i infer --model sd15 --prompt "a cat" --cuda-visible-devices 1
```

如果你遇到 `Tensor.item() cannot be called on meta tensors`（常见于 Janus/Janus-Pro + 新版 Transformers 的 meta 初始化路径），请更新到最新代码后重试；我们已在 runner 里强制禁用了 meta 初始化。

如果你遇到 `CVE-2025-32434` / `upgrade torch to at least v2.6` 相关报错（Janus-Pro 目前是 `.bin` 权重，会走 `torch.load`），请重装 Janus env：

```bash
cnm-t2i install --model janus-pro-1b --env-only --torch-backend cu121
```

如果你的 `cu121` 后端找不到 `torch>=2.6`，把后端换成 `cu124` 或 `auto` 再试。

## Notes

- Some models may be gated on Hugging Face (you must accept terms and provide a token).
- This repo intentionally keeps the CLI lightweight; heavy deps (torch/diffusers/etc) are installed into `cnm-t2i-work/envs/`.
