#!/usr/bin/env python3
"""
Generate DVD ABAW Test-Set submission CSVs for multiple trained models.

The script reuses inference helpers from `exp_backbones_v2.ipynb` to keep
behavior close to your training/eval pipeline.

Expected template format (from Expected_Output_Files):
  - columns: Frame_Number, Label
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def _resolve_path(value: Optional[Union[Path, str]], project_root: Path) -> Optional[Path]:
    if value is None:
        return None
    p = Path(value)
    if p.is_absolute():
        return p.resolve()
    return (project_root / p).resolve()


def _strip_notebook_magics(src: str) -> str:
    out = []
    for line in src.splitlines():
        t = line.lstrip()
        if t.startswith("!") or t.startswith("%"):
            continue
        out.append(line)
    return "\n".join(out)


def _patch_tqdm_behavior(g: dict, quiet_infer: bool = True, mininterval: float = 1.5) -> None:
    tq = g.get("tqdm")
    if tq is None:
        return

    def _wrapped(iterable=None, *args, **kwargs):
        desc = str(kwargs.get("desc", "")).strip().lower()
        kwargs.setdefault("dynamic_ncols", False)
        kwargs.setdefault("mininterval", float(mininterval))
        if quiet_infer and desc.startswith("infer"):
            kwargs["disable"] = True
        return tq(iterable, *args, **kwargs)

    g["tqdm"] = _wrapped


def _load_notebook_runtime(notebook_path: Path) -> dict:
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    g: dict = {"__name__": "__main__"}

    # Same minimal block as in run_full_val_eval_v2.py, minus full-val cell.
    cells_to_run = [6, 7, 8, 9, 10, 11, 12, 13, 29]
    for idx in cells_to_run:
        cell = nb["cells"][idx]
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        src = _strip_notebook_magics(src)
        exec(compile(src, f"cell_{idx}", "exec"), g, g)

    _patch_tqdm_behavior(g, quiet_infer=True, mininterval=1.5)
    return g


def _load_run_cfg(run_dir: Path) -> Tuple[dict, Optional[str]]:
    cfg_path = run_dir / "cfg_effective.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            if isinstance(cfg, dict):
                return cfg, f"cfg_effective:{cfg_path}"
        except Exception:
            pass

    full_val_path = run_dir / "full_val_result.json"
    if full_val_path.exists():
        try:
            j = json.loads(full_val_path.read_text(encoding="utf-8"))
            cfg = j.get("cfg_snapshot")
            if isinstance(cfg, dict):
                return cfg, f"full_val_result:{full_val_path}"
        except Exception:
            pass

    return {}, None


def _resolve_frame_step(
    explicit_frame_step: Optional[int],
    ckpt: dict,
    cfg: dict,
    run_dir: Path,
) -> Tuple[int, str]:
    if explicit_frame_step is not None:
        return int(explicit_frame_step), "model_spec"

    ckpt_cfg = ckpt.get("cfg") if isinstance(ckpt, dict) else None
    if isinstance(ckpt_cfg, dict) and ckpt_cfg.get("frame_step") is not None:
        return int(ckpt_cfg["frame_step"]), "checkpoint_cfg"

    hist = run_dir / "full_val_result.json"
    if hist.exists():
        try:
            j = json.loads(hist.read_text(encoding="utf-8"))
            if isinstance(j, dict) and j.get("frame_step") is not None:
                return int(j["frame_step"]), "historical_fullval"
        except Exception:
            pass

    return int(cfg.get("frame_step", 2)), "cfg_default"


def _load_model(project_root: Path, backbone: str, run_dir: Path, cfg: dict, device: str):
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from models import build_model  # pylint: disable=import-error
    import torch

    model = build_model(
        backbone=backbone,
        num_classes=2,
        pretrained=True,
        dropout=0.3,
        cfg=cfg,
    ).to(device)

    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    return model, ckpt, ckpt_path


def _predict_one_video(
    g: dict,
    model,
    device: str,
    video_id: str,
    video_path: Optional[Path],
    backend: str,
    clip_len: int,
    frame_step: int,
    stride: Optional[int],
    threshold: float,
    smooth_win: int,
    frames_root: Optional[Path] = None,
    flow_root: Optional[Path] = None,
    skeleton_root: Optional[Path] = None,
) -> np.ndarray:
    infer_fn = g["infer_video_per_frame"]
    smooth_fn = g.get("smooth_probs")

    probs, _ = infer_fn(
        model=model,
        video_path=video_path if backend != "frames" else None,
        video_id=video_id,
        frames_root=frames_root,
        ann_path=None,
        clip_len=int(clip_len),
        frame_step=int(frame_step),
        stride=None if stride is None else int(stride),
        threshold=float(threshold),
        transform=g.get("val_tfms"),
        backend=backend,
        device=device,
        debug=False,
        flow_frames_root=flow_root,
        skeleton_root=skeleton_root,
    )
    probs = np.asarray(probs, dtype=np.float32)
    if smooth_fn is not None and int(smooth_win) > 1 and probs.size > 0:
        probs = np.asarray(smooth_fn(probs, win=int(smooth_win)), dtype=np.float32)
    pred = (probs >= float(threshold)).astype(np.int64)
    return pred


def _write_submission_csv(template_csv: Optional[Path], out_csv: Path, pred: np.ndarray) -> None:
    if template_csv is not None and template_csv.exists():
        df = pd.read_csv(template_csv)
        if "Frame_Number" not in df.columns:
            raise ValueError(f"Template has no Frame_Number column: {template_csv}")

        frame_numbers = df["Frame_Number"].to_numpy()
        if pred.size == 0:
            labels = np.zeros((len(frame_numbers),), dtype=np.int64)
        else:
            idx = np.clip(frame_numbers.astype(np.int64) - 1, 0, pred.size - 1)
            labels = pred[idx]
        df["Label"] = labels.astype(np.int64)
    else:
        n = int(pred.size)
        frame_numbers = np.arange(1, n + 1, dtype=np.int64)
        labels = pred.astype(np.int64) if n > 0 else np.zeros((0,), dtype=np.int64)
        df = pd.DataFrame({"Frame_Number": frame_numbers, "Label": labels})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def _collect_video_ids(
    backend: str,
    templates_root: Optional[Path],
    videos_root: Optional[Path],
    frames_root: Optional[Path],
    video_ext: str,
    only_ids: Optional[List[str]] = None,
) -> List[str]:
    backend = str(backend).lower()
    if backend == "frames":
        if frames_root is None:
            raise ValueError("backend='frames' but frames_root is not provided.")
        if not frames_root.exists():
            raise FileNotFoundError(f"frames_root does not exist: {frames_root}")
        source_ids = {p.name for p in frames_root.iterdir() if p.is_dir()}
    else:
        if videos_root is None:
            raise ValueError(f"backend='{backend}' but videos_root is not provided.")
        if not videos_root.exists():
            raise FileNotFoundError(f"videos_root does not exist: {videos_root}")
        source_ids = {p.stem for p in videos_root.glob(f"*{video_ext}")}

    if templates_root is not None and templates_root.exists():
        template_ids = {p.stem for p in templates_root.glob("*.csv")}
        ids = template_ids & source_ids
    else:
        ids = source_ids
    if only_ids:
        ids = ids & set(only_ids)
    ids = sorted(ids)
    return ids


def _maybe_zip_dir(src_dir: Path, zip_path: Path) -> None:
    import zipfile

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(src_dir.glob("*.csv")):
            zf.write(p, arcname=p.name)


def run(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    notebook_path = _resolve_path(args.notebook, project_root)
    suite_path = _resolve_path(args.models_suite, project_root)
    templates_root = _resolve_path(args.templates_root, project_root)
    test_videos_root = _resolve_path(args.test_videos_root, project_root)
    out_root = _resolve_path(args.out_root, project_root)

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    models = suite.get("models", suite if isinstance(suite, list) else None)
    if not isinstance(models, list) or len(models) == 0:
        raise ValueError("models_suite must be a list or {'models': [...]} with at least 1 model.")
    if len(models) > 5:
        print(f"[warn] You provided {len(models)} models. Competition allows up to 5 submissions.")

    only_ids = None
    if args.video_ids:
        only_ids = [x.strip() for x in args.video_ids.split(",") if x.strip()]
    if templates_root is None or (not templates_root.exists()):
        print("[warn] templates_root missing -> writing CSV as Frame_Number=1..N from predictions")

    g = _load_notebook_runtime(notebook_path)
    device = str(args.device)
    summary = {"project_root": str(project_root), "models": []}

    for i, spec in enumerate(models, start=1):
        name = str(spec.get("name", f"model_{i}"))
        backbone = str(spec["backbone"])
        run_dir = _resolve_path(spec["run_dir"], project_root)
        threshold = float(spec.get("threshold", 0.5))
        smooth_win = int(spec.get("smooth_win", 1))
        clip_len_override = spec.get("clip_len")
        frame_step_override = spec.get("frame_step")
        stride = spec.get("stride")

        run_cfg, src = _load_run_cfg(run_dir)
        cfg = dict(g.get("CFG", {}))
        cfg.update(run_cfg)
        cfg["backbone"] = backbone
        if "backend" in spec:
            cfg["backend"] = str(spec["backend"])
        if clip_len_override is not None:
            cfg["clip_len"] = int(clip_len_override)

        backend = str(cfg.get("backend", "frames"))
        clip_len = int(cfg.get("clip_len", 16))

        model_frames_root = spec.get("test_frames_root", args.test_frames_root)
        model_flow_root = spec.get("test_flow_root", args.test_flow_root)
        model_skeleton_root = spec.get("test_skeleton_root", args.test_skeleton_root)

        model_frames_root = _resolve_path(model_frames_root, project_root)
        model_flow_root = _resolve_path(model_flow_root, project_root)
        model_skeleton_root = _resolve_path(model_skeleton_root, project_root)

        if backend == "frames" and model_frames_root is None:
            raise ValueError(f"[{name}] backend=frames but no test_frames_root provided.")

        model_video_ids = _collect_video_ids(
            backend=backend,
            templates_root=templates_root,
            videos_root=test_videos_root,
            frames_root=model_frames_root,
            video_ext=args.video_ext,
            only_ids=only_ids,
        )
        if len(model_video_ids) == 0:
            raise RuntimeError(
                f"[{name}] No video ids found for backend={backend}. "
                "Check test root paths and --video-ext."
            )

        model, ckpt, ckpt_path = _load_model(project_root, backbone, run_dir, cfg, device=device)
        frame_step, frame_step_src = _resolve_frame_step(frame_step_override, ckpt, cfg, run_dir)

        model_out_dir = out_root / name
        model_out_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[{name}] backbone={backbone} run={run_dir.name} ckpt={ckpt_path.name} "
            f"backend={backend} clip_len={clip_len} frame_step={frame_step}({frame_step_src}) "
            f"threshold={threshold} smooth_win={smooth_win}"
        )
        print(f"[{name}] videos to process: {len(model_video_ids)}")
        if src:
            print(f"[{name}] cfg source: {src}")

        errors = []
        for vi, video_id in enumerate(model_video_ids, start=1):
            video_path = (
                (test_videos_root / f"{video_id}{args.video_ext}")
                if backend != "frames"
                else None
            )
            template_csv = (templates_root / f"{video_id}.csv") if templates_root is not None else None
            out_csv = model_out_dir / f"{video_id}.csv"
            try:
                pred = _predict_one_video(
                    g=g,
                    model=model,
                    device=device,
                    video_id=video_id,
                    video_path=video_path,
                    backend=backend,
                    clip_len=clip_len,
                    frame_step=frame_step,
                    stride=None if stride is None else int(stride),
                    threshold=threshold,
                    smooth_win=smooth_win,
                    frames_root=model_frames_root,
                    flow_root=model_flow_root,
                    skeleton_root=model_skeleton_root,
                )
                _write_submission_csv(template_csv, out_csv, pred)
                if vi % 5 == 0 or vi == len(model_video_ids):
                    print(f"[{name}] {vi}/{len(model_video_ids)} done")
            except Exception as e:  # pylint: disable=broad-except
                msg = f"{video_id}: {type(e).__name__}: {e}"
                print(f"[{name}][error] {msg}")
                errors.append(msg)

        zip_path = out_root / f"{name}.zip"
        if args.zip:
            _maybe_zip_dir(model_out_dir, zip_path)
            print(f"[{name}] zip: {zip_path}")

        model_summary = {
            "name": name,
            "backbone": backbone,
            "run_dir": str(run_dir),
            "checkpoint": str(ckpt_path),
            "threshold": threshold,
            "smooth_win": smooth_win,
            "clip_len": clip_len,
            "frame_step": frame_step,
            "frame_step_source": frame_step_src,
            "backend": backend,
            "out_dir": str(model_out_dir),
            "zip_path": str(zip_path) if args.zip else None,
            "n_videos": len(model_video_ids),
            "n_errors": len(errors),
            "errors": errors,
        }
        summary["models"].append(model_summary)

    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "submission_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate test-set submission CSVs for multiple models.")
    ap.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="Path to project with models.py and runs/.",
    )
    ap.add_argument(
        "--notebook",
        type=Path,
        required=True,
        help="Path to exp_backbones_v2.ipynb (used to reuse helper inference functions).",
    )
    ap.add_argument(
        "--models-suite",
        type=Path,
        required=True,
        help="JSON file with list of models ({'models': [...]}).",
    )
    ap.add_argument(
        "--test-videos-root",
        type=Path,
        default=None,
        help="Folder with test .mp4 files. Can be absolute or relative to --project-root.",
    )
    ap.add_argument(
        "--templates-root",
        type=Path,
        default=None,
        help="Folder with expected submission templates (*.csv). Absolute or relative to --project-root.",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("test_set_processing/submissions_generated"),
        help="Where to write per-model submission folders. Absolute or relative to --project-root.",
    )
    ap.add_argument(
        "--video-ext",
        type=str,
        default=".mp4",
        help="Video extension in test-videos-root.",
    )
    ap.add_argument(
        "--video-ids",
        type=str,
        default=None,
        help="Optional comma-separated subset of video ids, e.g. 0008,0012",
    )
    ap.add_argument(
        "--test-frames-root",
        type=Path,
        default=None,
        help="Optional root with extracted test frames: <root>/<video_id>/*.jpg. Absolute or relative to --project-root.",
    )
    ap.add_argument(
        "--test-flow-root",
        type=Path,
        default=None,
        help="Optional root with precomputed flow frames: <root>/<video_id>/*.jpg. Absolute or relative to --project-root.",
    )
    ap.add_argument(
        "--test-skeleton-root",
        type=Path,
        default=None,
        help="Optional root with precomputed skeleton features: <root>/<video_id>.npy. Absolute or relative to --project-root.",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Inference device (cuda/cpu).",
    )
    ap.add_argument(
        "--zip",
        action="store_true",
        help="Also create one zip archive per model output folder.",
    )
    return ap.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[fatal] {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
