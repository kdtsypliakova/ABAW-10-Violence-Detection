# Test Set Processing

Папка для генерации submission CSV/ZIP на тесте.

## Что за что отвечает

- `generate_submissions.py` — основной скрипт инференса.
- `run_all.sh` — короткий launcher (обертка над `generate_submissions.py`).
- `models_suite*.json` — набор моделей для прогона (1..5).
- `submissions_*` — результаты (`<model_name>/*.csv` + опционально `<model_name>.zip`).
- `submission_summary.json` — сводка по запуску.

## Правила по путям

- В `models_suite*.json` пути можно писать:
  - абсолютные (`/home/...`)
  - относительные от `--project-root` (`runs/...`, `skeleton/Test`).
- В скрипте это уже нормализовано.
- Если `templates_root` не существует, CSV генерятся как `Frame_Number=1..N`.

## Важный anti-footgun

Если в ноутбуке строишь пути через `ROOT / CFG.get(...)`, не ставь `/` в начале значения.

Правильно:

```python
TRAIN_FLOW_FRAMES = ROOT / CFG.get("flow_frames_root_train", "flow/Training")
VAL_FLOW_FRAMES   = ROOT / CFG.get("flow_frames_root_val", "flow/Validation")
```

Неправильно:

```python
TRAIN_FLOW_FRAMES = ROOT / CFG.get("flow_frames_root_train", "/flow/Training")
```

(второй вариант отбросит `ROOT`).

## Пример models_suite (1 модель)

```json
{
  "models": [
    {
      "name": "convnext_bilstm_unimodal",
      "backbone": "convnext_tiny_bilstm",
      "run_dir": "runs/convnext_tiny_clip32_0.78",
      "threshold": 0.5,
      "smooth_win": 1,
      "frame_step": 1,
      "backend": "frames"
    }
  ]
}
```

## Полный запуск (все видео)

```bash
PROJECT_ROOT="$(pwd)"

python3 "$PROJECT_ROOT/test_set_processing/generate_submissions.py" \
  --project-root "$PROJECT_ROOT" \
  --notebook "exp_backbones_v2.ipynb" \
  --models-suite "test_set_processing/models_suite.example.json" \
  --test-frames-root "/path/to/DVD/Test/frames" \
  --templates-root "__NO_TEMPLATES__" \
  --out-root "test_set_processing/submissions_generated" \
  --device cuda \
  --zip
```

Альтернатива через launcher:

```bash
bash test_set_processing/run_all.sh \
  --models-suite test_set_processing/models_suite.example.json \
  --test-frames-root /path/to/DVD/Test/frames
```

## Smoke test (1-2 видео)

```bash
PROJECT_ROOT="$(pwd)"

python3 "$PROJECT_ROOT/test_set_processing/generate_submissions.py" \
  --project-root "$PROJECT_ROOT" \
  --notebook "exp_backbones_v2.ipynb" \
  --models-suite "test_set_processing/models_suite.example.json" \
  --test-frames-root "/path/to/DVD/Test/frames" \
  --video-ids 0008,0012 \
  --out-root "test_set_processing/submissions_smoke" \
  --device cuda
```

## Формат модели в suite

Для каждой модели:

- `name` — имя папки/zip.
- `backbone` — имя бэкбона.
- `run_dir` — путь к run с `best.pt`.
- Опционально: `threshold`, `smooth_win`, `frame_step`, `stride`, `clip_len`, `backend`.
- Опционально roots для модальностей:
  - `test_frames_root`
  - `test_flow_root`
  - `test_skeleton_root`
