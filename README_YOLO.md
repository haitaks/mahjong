# Развёртывание пайплайна на другой машине

На этой машине хранятся исходники `mahjong_layout` и тестовые фото.
На целевой машине (с GPU) запускаются: обучение YOLO + прогон на фото.

---

## Шаг 1 — скопировать проект и датасет на целевую машину

На этой машине (источник):

```bash
cd /home/rosalvak/projects/mahjong

# Упаковать датасет
tar czf yolo_dataset.tar.gz yolo/

# Отправить на целевую машину
scp yolo_dataset.tar.gz user@target-machine:~/mahjong/

# (Опционально) отправить исходники, если клона репы нет
scp -r . user@target-machine:~/mahjong/
```

На целевой машине:

```bash
cd ~/mahjong
tar xzf yolo_dataset.tar.gz    # появится yolo/ с train/valid/test/ + data.yaml
```

---

## Шаг 2 — установить зависимости

На целевой машине:

```bash
cd ~/mahjong
pip install -e .                     # mahjong_layout (numpy + pillow)
pip install ultralytics              # YOLO
pip install opencv-python            # для overlay в run_pipeline.py

# (Опционально) OCR для wan-иероглифов
pip install -e ".[ocr]"
```

---

## Шаг 3 — обучить YOLO

```bash
cd ~/mahjong

# YOLOv8n (быстрая, ~2-4 часа на среднем GPU)
python scripts/train_yolo.py

# YOLOv8s (точнее, дольше)
python scripts/train_yolo.py --model yolov8s

# На CPU
python scripts/train_yolo.py --device cpu --epochs 50 --batch 4
```

Параметры:

| Флаг          | Умолч. | Описание                                    |
|---------------|--------|---------------------------------------------|
| `--model`     | yolov8n| Вариант: yolov8n/s/m/l/x                    |
| `--epochs`    | 100    | Эпох обучения                               |
| `--batch`     | 8      | Размер батча (4 GB VRAM → 4, 8 GB → 8)     |
| `--device`    | cuda   | `cuda`, `cuda:0`, `cpu`                     |

После обучения веса: `yolo/runs/detect/train/weights/best.pt`

---

## Шаг 4 — запустить пайплайн на тестовых фото

```bash
cd ~/mahjong

# Все фото из tests/*.jpg (авто-ищет yolo/model.pt или yolo/runs/.../best.pt)
python scripts/run_pipeline.py

# Конкретные фото
python scripts/run_pipeline.py --photos tests/photo1.jpg tests/photo2.jpg

# Если веса ещё не скопированы в yolo/model.pt — указать путь
python scripts/run_pipeline.py --weights yolo/runs/detect/train/weights/best.pt

# Только детекция YOLO (без классификации)
python scripts/run_pipeline.py --no-classify
```

Результаты: `out/yolo_pipeline/*_pipeline.jpg`

---

## Шаг 5 — (опционально) скопировать веса обратно на эту машину

На целевой:

```bash
scp yolo/runs/detect/train/weights/best.pt \
  user@this-machine:/home/rosalvak/projects/mahjong/yolo/model.pt
```

---

## Структура проекта

```
mahjong/
├── mahjong_layout/              # Пакет классификации и зонирования (pip install -e .)
├── yolo/                        # Датасет (YOLO-формат)
│   ├── data.yaml
│   ├── train/images/            # 158 фото
│   ├── valid/images/            # 50 фото
│   ├── test/images/             # 10 фото
│   ├── model.pt                 ← веса YOLO (после обучения/копирования)
│   └── runs/                    ← результаты обучения (игнорируется git)
├── scripts/
│   ├── train_yolo.py            # Обучение YOLO
│   └── run_pipeline.py          # YOLO → classify_layout → overlay
├── tests/*.jpg                  # Боевые фото для теста
├── pyproject.toml               # pip install -e .
├── README.md
└── README_YOLO.md               ← Этот файл
```

**Важно:** YOLO детектит все тайлы (координаты боксов). Классификацию по мастям
(wan/pin/tiao/honor + значение) делает `classify_tile` на кропе.
