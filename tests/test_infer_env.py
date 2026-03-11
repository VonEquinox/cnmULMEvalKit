from __future__ import annotations

from pathlib import Path

from cnm_t2i import infer, paths, state, uvwrap


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


def test_infer_passes_hf_endpoint_and_proxy_from_state(tmp_path: Path, monkeypatch) -> None:
    captured = {"env": None}

    def fake_run_python(*args, **kwargs):
        captured["env"] = kwargs.get("env")
        return uvwrap.CommandResult(cmd=["python"], env=captured["env"])

    monkeypatch.setattr(uvwrap, "run_python", fake_run_python)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("http_proxy", raising=False)
    monkeypatch.delenv("https_proxy", raising=False)

    home = tmp_path / "home"
    layout = paths.HomeLayout.from_home(home)
    rec = state.ModelInstallRecord(
        key="janus-pro-1b",
        hf_repo="deepseek-ai/Janus-Pro-1B",
        runtime="janus",
        torch_backend="cu121",
        python="3.11",
        env_dir=str(tmp_path / "env"),
        model_dir=str(tmp_path / "model"),
        hf_endpoint="https://hf-mirror.com",
        proxy="http://127.0.0.1:7897",
    )
    st = state.AppState(installs={"janus-pro-1b": rec})
    state.save_state(layout.state_path, st)

    req = infer.InferRequest(
        home=home,
        model_key="janus-pro-1b",
        model_path=None,
        env_dir=None,
        prompt="a cat",
        out=tmp_path / "out.png",
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
        cuda_visible_devices=None,
        cfg_weight=5.0,
        parallel_size=1,
        dry_run=True,
    )
    _ = infer.infer(req)
    assert captured["env"] is not None
    assert captured["env"]["HF_ENDPOINT"] == "https://hf-mirror.com"
    assert captured["env"]["HTTPS_PROXY"] == "http://127.0.0.1:7897"
