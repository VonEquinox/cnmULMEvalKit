from __future__ import annotations


def diffusers_runner_script() -> str:
    # Standalone script executed inside the runtime venv.
    return r'''#!/usr/bin/env python3
import argparse
import inspect
from pathlib import Path


def _pick_device(device: str):
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _pick_dtype(dtype: str, device: str):
    import torch

    if dtype == "fp32":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    # auto
    if device != "cuda":
        return torch.float32
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


def _filter_kwargs(fn, kwargs):
    try:
        sig = inspect.signature(fn)
    except Exception:
        return kwargs
    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return kwargs
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed and v is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--family", default="auto", choices=["auto", "stable-diffusion", "flux", "qwen-image"])
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--true-cfg-scale", type=float, default=4.0)
    ap.add_argument("--max-seq-len", type=int, default=256)
    args = ap.parse_args()

    import torch
    from diffusers import DiffusionPipeline

    device = _pick_device(args.device)
    torch_dtype = _pick_dtype(args.dtype, device)

    model_path = args.model_path
    family = args.family

    pipe_cls = None
    if family == "qwen-image":
        try:
            from diffusers import QwenImagePipeline  # type: ignore
            pipe_cls = QwenImagePipeline
        except Exception:
            pipe_cls = None
    elif family == "flux":
        try:
            from diffusers import FluxPipeline  # type: ignore
            pipe_cls = FluxPipeline
        except Exception:
            pipe_cls = None

    if pipe_cls is None:
        pipe_cls = DiffusionPipeline

    pipe = pipe_cls.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        use_safetensors=True,
    )
    pipe.to(device)
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass

    gen = torch.Generator(device=device).manual_seed(args.seed)
    call_kwargs = {
        "prompt": args.prompt,
        "negative_prompt": args.negative,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance,
        "generator": gen,
        # Qwen-Image specific
        "true_cfg_scale": args.true_cfg_scale,
        # Flux specific
        "max_sequence_length": args.max_seq_len,
        "output_type": "pil",
    }
    call_kwargs = _filter_kwargs(pipe.__call__, call_kwargs)
    result = pipe(**call_kwargs)
    images = getattr(result, "images", None)
    if images is None:
        raise RuntimeError("Pipeline did not return .images")
    if not images:
        raise RuntimeError("Pipeline returned empty images")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(out_path)


if __name__ == "__main__":
    main()
'''


def janus_runner_script() -> str:
    # Standalone script executed inside the runtime venv.
    # Supports Janus / Janus-Pro (autoregressive) and JanusFlow (rectified flow).
    return r'''#!/usr/bin/env python3
import argparse
import math
from pathlib import Path


def _pick_device(device: str):
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _pick_dtype(dtype: str, device: str):
    import torch

    if dtype == "fp32":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    # auto
    if device != "cuda":
        return torch.float32
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


def _seed_all(seed: int):
    import torch
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_first(decoded_images, out_path: Path):
    import numpy as np
    from PIL import Image

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(decoded_images, "detach"):
        arr = decoded_images.detach().cpu().numpy()
        if arr.ndim == 4:
            arr = arr.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        img = (arr[0] * 255).astype(np.uint8)
        Image.fromarray(img).save(out_path)
        return
    raise RuntimeError("Unexpected decoded_images type")


def run_janus_autoregressive(*, model_path: str, prompt: str, out: str, family: str, seed: int, device: str, dtype: str, temperature: float, cfg_weight: float, parallel_size: int):
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM
    from janus.models import VLChatProcessor  # from Janus GitHub package

    _seed_all(seed)

    device = _pick_device(device)
    torch_dtype = _pick_dtype(dtype, device)
    if device != "cuda":
        # Janus image generation is intended for GPU; CPU will be extremely slow / may OOM.
        raise RuntimeError("Janus image generation requires CUDA for practical use.")

    if family == "janus-pro":
        user_role = "<|User|>"
        assistant_role = "<|Assistant|>"
        sft_format = "janus-pro"
    else:
        user_role = "User"
        assistant_role = "Assistant"
        sft_format = "chatml"

    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    # Newer Transformers may default to a low-CPU-memory load path that instantiates
    # modules on the "meta" device. Janus's SigLIP ViT init calls Tensor.item(),
    # which fails on meta tensors. Force a normal load here.
    vl_gpt = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
    )
    vl_gpt = vl_gpt.cuda().eval()

    conversation = [
        {"role": user_role, "content": prompt},
        {"role": assistant_role, "content": ""},
    ]

    sft = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=sft_format,
        system_prompt="",
    )
    sft = sft + vl_chat_processor.image_start_tag

    inputs = vl_chat_processor(
        sft,
        images=None,
        force_batchify=True,
    ).to(vl_gpt.device)

    input_ids = inputs["input_ids"]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    # Image token config (fixed for Janus image VQ).
    img_size = 384
    patch_size = 16
    image_token_num_per_image = int((img_size / patch_size) ** 2)

    tokens = torch.zeros((parallel_size * 2, input_ids.shape[1]), dtype=torch.int).to(vl_gpt.device)
    tokens[:, :] = input_ids[0, :]
    tokens[parallel_size:, 1:] = pad_id

    inputs_embeds = vl_gpt.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(vl_gpt.device)

    past_key_values = None
    for i in range(image_token_num_per_image):
        outputs = vl_gpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=past_key_values,
        )
        past_key_values = outputs.past_key_values
        hidden_states = outputs.last_hidden_state
        logits = vl_gpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0:parallel_size, :]
        logit_uncond = logits[parallel_size:, :]
        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(dim=-1)
        generated_tokens[:, i] = next_token

        # Duplicate for conditional/unconditional streams.
        next_token = torch.cat([next_token, next_token], dim=0)
        img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(1)

    decoded_images = vl_gpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size],
    )
    decoded_images = decoded_images.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    decoded_images = (decoded_images * 255).astype(np.uint8)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    Image.fromarray(decoded_images[0]).save(out_path)


def run_janusflow(*, model_path: str, prompt: str, out: str, seed: int, device: str, dtype: str, cfg_weight: float, steps: int):
    import torch
    import numpy as np
    from PIL import Image
    from janus.janusflow.models import MultiModalityCausalLM, VLChatProcessor
    from diffusers.models import AutoencoderKL

    _seed_all(seed)

    device = _pick_device(device)
    if device != "cuda":
        raise RuntimeError("JanusFlow generation requires CUDA.")

    # JanusFlow uses an SDXL VAE that does not work reliably with fp16.
    if dtype == "fp16":
        raise RuntimeError("JanusFlow does not support fp16; use bf16 or fp32.")

    torch_dtype = _pick_dtype(dtype, device)
    if torch_dtype == torch.float16:
        # Covers dtype=auto on GPUs without bf16 support.
        raise RuntimeError("JanusFlow auto dtype selected fp16; please use --dtype bf16 or --dtype fp32.")

    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt = MultiModalityCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    ).to(torch_dtype).cuda().eval()

    # VAE used by the official JanusFlow demo (downloaded via HF if not cached).
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
    vae = vae.to(torch_dtype).cuda().eval()

    messages = [
        {"role": "User", "content": prompt},
        {"role": "Assistant", "content": ""},
    ]
    sft = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=messages,
        sft_format=getattr(vl_chat_processor, "sft_format", "chatml"),
        system_prompt="",
    )
    image_tag = getattr(vl_chat_processor, "image_gen_tag", None) or getattr(
        vl_chat_processor, "image_start_tag"
    )
    if not image_tag:
        raise RuntimeError("JanusFlow processor did not expose image_gen_tag/image_start_tag.")
    text = sft + image_tag

    input_ids = tokenizer.encode(text)
    input_ids = torch.LongTensor(input_ids)

    # One image output (CFG uses a conditional+unconditional pair).
    batchsize = 1
    tokens = torch.stack([input_ids] * (2 * batchsize)).cuda()
    pad_id = getattr(vl_chat_processor, "pad_id", None)
    if pad_id is None:
        pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    tokens[batchsize:, 1:] = int(pad_id)
    inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)

    # Remove the last token (as in the upstream demo) to insert t_emb later.
    inputs_embeds = inputs_embeds[:, :-1, :]

    z = torch.randn((batchsize, 4, 48, 48), dtype=torch_dtype).cuda()
    dt = torch.zeros_like(z).cuda().to(torch_dtype) + (1.0 / float(steps))

    attention_mask = torch.ones(
        (2 * batchsize, inputs_embeds.shape[1] + 577), device=vl_gpt.device
    )
    attention_mask[batchsize:, 1:inputs_embeds.shape[1]] = 0
    attention_mask = attention_mask.int()

    for step in range(int(steps)):
        z_input = torch.cat([z, z], dim=0)  # for CFG
        t = step / float(steps) * 1000.0
        t = torch.tensor([t] * z_input.shape[0], device=z_input.device).to(dt)
        z_enc = vl_gpt.vision_gen_enc_model(z_input, t)
        z_emb, t_emb, hs = z_enc[0], z_enc[1], z_enc[2]
        z_emb = z_emb.view(z_emb.shape[0], z_emb.shape[1], -1).permute(0, 2, 1)
        z_emb = vl_gpt.vision_gen_enc_aligner(z_emb)

        llm_emb = torch.cat([inputs_embeds, t_emb.unsqueeze(1), z_emb], dim=1)
        outputs = vl_gpt.language_model.model(
            inputs_embeds=llm_emb,
            use_cache=False,
            attention_mask=attention_mask,
            past_key_values=None,
        )
        hidden_states = outputs.last_hidden_state
        hidden_states = vl_gpt.vision_gen_dec_aligner(
            vl_gpt.vision_gen_dec_aligner_norm(hidden_states[:, -576:, :])
        )
        hidden_states = hidden_states.reshape(z_emb.shape[0], 24, 24, 768).permute(0, 3, 1, 2)
        v = vl_gpt.vision_gen_dec_model(hidden_states, hs, t_emb)
        v_cond, v_uncond = torch.chunk(v, 2)
        v = cfg_weight * v_cond - (cfg_weight - 1.0) * v_uncond
        z = z + dt * v

    decoded = vae.decode(z / vae.config.scaling_factor).sample
    img = decoded[0].float().clamp(-1.0, 1.0) * 0.5 + 0.5
    img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--family", required=True, choices=["janus", "janus-pro", "janusflow"])
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--cfg-weight", type=float, default=5.0)
    ap.add_argument("--parallel-size", type=int, default=1)
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()

    if args.family == "janusflow":
        run_janusflow(
            model_path=args.model_path,
            prompt=args.prompt,
            out=args.out,
            seed=args.seed,
            device=args.device,
            dtype=args.dtype,
            cfg_weight=args.cfg_weight,
            steps=args.steps,
        )
    else:
        run_janus_autoregressive(
            model_path=args.model_path,
            prompt=args.prompt,
            out=args.out,
            family=args.family,
            seed=args.seed,
            device=args.device,
            dtype=args.dtype,
            temperature=args.temperature,
            cfg_weight=args.cfg_weight,
            parallel_size=args.parallel_size,
        )


if __name__ == "__main__":
    main()
'''
