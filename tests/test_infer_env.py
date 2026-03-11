from __future__ import annotations

from pathlib import Path

from cnm_t2i import infer, uvwrap


def test_infer_passes_cuda_visible_devices_env(tmp_path: Path, monkeypatch) -> None:
    captured = {"env": None}

    def fake_run_python(*args, **kwargs):
        captured["env"] = kwargs.get("env")
        return uvwrap.CommandResult(cmd=["python"], env=captured["env"])

    monkeypatch.setattr(uvwrap, "run_python", fake_run_python)

    out_path = tmp_path / "out.png"
    req = infer.InferRequest(
        home=tmp_path / "home",
        model_key=None,
        model_path=tmp_path / "model",
        env_dir=tmp_path / "env",
        prompt="a cat",
        out=out_path,
        seed=0,
        steps=1,
        height=64,
        width=64,
        guidance=1.0,
        negative=None,
        true_cfg_scale=4.0,
        max_seq_len=128,
        dtype="auto",
        device="auto",
        cuda_visible_devices="1",
        cfg_weight=5.0,
        parallel_size=1,
        dry_run=True,
    )
    res = infer.infer(req)
    assert res == out_path
    assert captured["env"] is not None
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "1"

