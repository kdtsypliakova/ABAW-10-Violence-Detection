import argparse
import json
from pathlib import Path


def _jsonable(x):
    if hasattr(x, 'tolist'):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    return x


def _resolve_frame_step(cli_frame_step, ckpt, cfg, run_dir=None):
    if cli_frame_step is not None:
        return int(cli_frame_step), 'cli'

    ckpt_cfg = ckpt.get('cfg') if isinstance(ckpt, dict) else None
    if isinstance(ckpt_cfg, dict) and ckpt_cfg.get('frame_step') is not None:
        return int(ckpt_cfg['frame_step']), 'checkpoint_cfg'

    # Prefer historical full-val settings for reproducibility if present.
    try:
        if run_dir is not None:
            hist = Path(run_dir) / 'full_val_result.json'
            if hist.exists():
                j = json.loads(hist.read_text(encoding='utf-8'))
                if isinstance(j, dict) and j.get('frame_step') is not None:
                    return int(j['frame_step']), 'historical_fullval'
    except Exception:
        pass

    return int(cfg.get('frame_step', 2)), 'runtime_cfg'


def _strip_notebook_magics(src: str) -> str:
    cleaned = []
    for line in src.splitlines():
        t = line.lstrip()
        if t.startswith('!') or t.startswith('%'):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _patch_tqdm_behavior(g, quiet_infer=True, mininterval=1.5):
    tq = g.get('tqdm')
    if tq is None:
        return

    def _wrapped(iterable=None, *args, **kwargs):
        desc = str(kwargs.get('desc', '')).strip().lower()
        kwargs.setdefault('dynamic_ncols', False)
        kwargs.setdefault('mininterval', float(mininterval))
        if quiet_infer and desc.startswith('infer'):
            kwargs['disable'] = True
        return tq(iterable, *args, **kwargs)

    g['tqdm'] = _wrapped


def _load_run_cfg(run_dir):
    run_dir = Path(run_dir)

    cfg_path = run_dir / 'cfg_effective.json'
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
            if isinstance(cfg, dict):
                return cfg, f'cfg_effective:{cfg_path}'
        except Exception:
            pass

    fv_path = run_dir / 'full_val_result.json'
    if fv_path.exists():
        try:
            j = json.loads(fv_path.read_text(encoding='utf-8'))
            cfg = j.get('cfg_snapshot')
            if isinstance(cfg, dict):
                return cfg, f'full_val_result:{fv_path}'
        except Exception:
            pass

    return {}, None


def _refresh_cfg_globals(g):
    cfg = g['CFG']
    root = Path(cfg['root'])
    g['ROOT'] = root

    g['TRAIN_VIDEOS'] = root / cfg['train_videos']
    g['TRAIN_ANNS'] = root / cfg['train_anns']
    g['TRAIN_FRAMES'] = root / cfg['train_frames']
    g['VAL_VIDEOS'] = root / cfg['val_videos']
    g['VAL_ANNS'] = root / cfg['val_anns']
    g['VAL_FRAMES'] = root / cfg['val_frames']

    g['TRAIN_FLOW_FRAMES'] = Path(cfg.get('flow_frames_root_train', 'flow/Training'))
    g['VAL_FLOW_FRAMES'] = Path(cfg.get('flow_frames_root_val', 'flow/Validation'))
    g['USE_FLOW'] = bool(cfg.get('use_flow', False))

    g['TRAIN_SKELETON'] = Path(cfg.get('skeleton_root_train', str(root / 'Training/skeleton')))
    g['VAL_SKELETON'] = Path(cfg.get('skeleton_root_val', str(root / 'Validation/skeleton')))
    g['USE_SKELETON'] = bool(cfg.get('use_skeleton', False))


def run_full_val(notebook_path, run_dir, backbone, threshold, smooth_win,
                 frame_step, stride, out_path, quiet_infer=True, tqdm_mininterval=1.5):
    nb = json.loads(Path(notebook_path).read_text(encoding='utf-8'))
    g = {'__name__': '__main__'}

    # Minimal required cells from notebook:
    # imports/config/utils/csv decode/transforms/dataset + inference + full-val eval
    cells_to_run = [6, 7, 8, 9, 10, 11, 12, 13, 29, 31]

    run_cfg, run_cfg_source = _load_run_cfg(run_dir)
    if run_cfg_source:
        print(f'[full-val] loaded run cfg from {run_cfg_source}')
    else:
        print('[full-val][warn] run cfg not found; using notebook defaults')

    g['run_dir'] = Path(run_dir)

    for idx in cells_to_run:
        cell = nb['cells'][idx]
        if cell.get('cell_type') != 'code':
            continue

        src = ''.join(cell.get('source', []))
        src = _strip_notebook_magics(src)
        exec(compile(src, f'cell_{idx}', 'exec'), g, g)

        # Cell 7 defines CFG and path globals. Align to run cfg before later cells.
        if idx == 7:
            if run_cfg:
                g['CFG'].update(run_cfg)
            g['CFG']['backbone'] = backbone
            _refresh_cfg_globals(g)
            print('[full-val] effective cfg:', {
                k: g['CFG'].get(k)
                for k in ['backbone', 'backend', 'clip_len', 'frame_step', 'use_flow', 'use_skeleton']
            })

    _patch_tqdm_behavior(g, quiet_infer=quiet_infer, mininterval=tqdm_mininterval)

    device = g.get('device', 'cuda')
    from models import build_model
    CFG = g['CFG']

    model = build_model(
        backbone=backbone,
        num_classes=2,
        pretrained=True,
        dropout=0.3,
        cfg=CFG,
    ).to(device)

    best_ckpt = Path(run_dir) / 'best.pt'
    if not best_ckpt.exists():
        raise FileNotFoundError(f'Not found: {best_ckpt}')

    ckpt = g['torch'].load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.to(device).eval()

    eval_frame_step, fs_source = _resolve_frame_step(frame_step, ckpt, CFG, run_dir=run_dir)
    eval_stride = None if stride is None else int(stride)

    print(f'[full-val] Loaded: {best_ckpt} (best_f1={ckpt.get("best_f1")})')
    print(f'[full-val] frame_step={eval_frame_step} ({fs_source}), stride={eval_stride}')

    res = g['eval_full_val_videos'](
        model=model,
        val_meta=g['val_meta'],
        videos_root=Path(g['VAL_VIDEOS']),
        frames_root=Path(g['VAL_FRAMES']),
        video_ext=CFG.get('video_ext', '.mp4'),
        threshold=float(threshold),
        smooth_win=int(smooth_win),
        clip_len=int(CFG.get('clip_len', 16)),
        frame_step=int(eval_frame_step),
        stride=eval_stride,
        transform=g.get('val_tfms'),
        backend=CFG.get('backend', 'frames'),
        device=device,
        verbose=True,
        flow_frames_root=g.get('VAL_FLOW_FRAMES') if bool(CFG.get('use_flow', False)) else None,
        skeleton_root=g.get('VAL_SKELETON') if bool(CFG.get('use_skeleton', False)) else None,
    )

    out = {
        'run_dir': str(run_dir),
        'backbone': backbone,
        'threshold': float(threshold),
        'smooth_win': int(smooth_win),
        'frame_step_used': int(eval_frame_step),
        'frame_step_source': fs_source,
        'stride_used': eval_stride,
        'result': _jsonable(res),
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')

    print('=== FULL VAL ===')
    print(f"  macro_f1:        {res['macro_f1']:.4f}")
    print(f"  f1_non_violence: {res['f1_non_violence']:.4f}")
    print(f"  f1_violence:     {res['f1_violence']:.4f}")
    print(f'  saved: {out_path}')


def main():
    from models import ALL_BACKBONES

    ap = argparse.ArgumentParser()
    ap.add_argument('--notebook', default='exp_backbones_v2.ipynb')
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--backbone', required=True, choices=ALL_BACKBONES)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--smooth-win', type=int, default=1)
    ap.add_argument('--frame-step', type=int, default=None)
    ap.add_argument('--stride', type=int, default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--show-infer-progress', action='store_true')
    ap.add_argument('--tqdm-mininterval', type=float, default=1.5)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else Path('full_val_logs') / f'{run_dir.name}_fullval.json'

    run_full_val(
        notebook_path=args.notebook,
        run_dir=run_dir,
        backbone=args.backbone,
        threshold=args.threshold,
        smooth_win=args.smooth_win,
        frame_step=args.frame_step,
        stride=args.stride,
        out_path=out,
        quiet_infer=not args.show_infer_progress,
        tqdm_mininterval=args.tqdm_mininterval,
    )


if __name__ == '__main__':
    main()
