# Развёртывание YOLO-пайплайна на другой машине

Этот гайд — для машины с GPU (или мощным CPU), на которой будет
обучаться YOLO. После обучения веса (model.pt) копируются обратно.

---

## Шаг 1 — перенести проект и датасет

На этой машине (источник):

```bash
cd /home/rosalvak/projects/mahjong

# 1. Упаковать датасет
tar czf yolo_dataset.tar.gz yolo/

# 2. Отправить на целевую машину
scp yolo_dataset.tar.gz user@target-machine:~/mahjong/

# 3. (Опционально) отправить исходники, если их нет на целевой
# (на другой ветке / свежий clone)
git push && ssh user@target-machine "git clone git@github.com:user/mahjong-layout.git ~/mahjong"
```

На целевой машине:

```bash
cd ~/mahjong
tar xzf yolo_dataset.tar.gz
# Появится папка yolo/ с train/valid/test/ + data.yaml
```

---

## Шаг 2 — установить зависимости

```bash
# Базовые — для mahjong_layout (классификация, пайплайн)
pip install -e .

# Для YOLO
pip install ultralytics

# (Опционально) OCR для распознавания wan-иероглифов
pip install -e ".[ocr]"
```

---

## Шаг 3 — обучить YOLO

```bash
cd ~/mahjong/yolo

# YOLOv8n — лёгкая (быстро, ~2-4 часа на среднем GPU)
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='data.yaml', epochs=100, imgsz=640, batch=8, device='cuda')
"

# YOLOv8s — точнее, но медленнее
# model = YOLO('yolov8s.pt')
```

Параметры под ваше железо:

| Параметр  | 4 GB VRAM | 6-8 GB VRAM | 12+ GB VRAM | CPU |
|-----------|-----------|-------------|-------------|-----|
| batch     | 4         | 8           | 16          | 2   |
| epochs    | 100       | 100         | 100         | 50  |
| device    | cuda:0    | cuda:0      | cuda:0      | cpu |

После обучения веса: `runs/detect/train/weights/best.pt`

---

## Шаг 4 — скопировать веса обратно

```bash
scp ~/mahjong/yolo/runs/detect/train/weights/best.pt \
  user@this-machine:/home/rosalvak/projects/mahjong/yolo/model.pt
```

---

## Шаг 5 — полный прогон на этой машине

```bash
cd /home/rosalvak/projects/mahjong

# Активировать venv (если используете)
source .venv/bin/activate

# Прогнать YOLO -> classify_layout на тестовых фото
python scripts/yolo_pipeline.py

# Результаты: out/yolo_pipeline/*_yolo_pipeline.jpg
```

---

## Структура проекта

```
mahjong/
├── yolo/                        # Датасет (YOLO-формат)
│   ├── data.yaml                # Конфиг датасета
│   ├── train/images/            # Тренировочные фото (158 шт)
│   ├── train/labels/            # YOLO-разметка
│   ├── valid/images/            # Валидационные фото (50 шт)
│   ├── valid/labels/            # YOLO-разметка
│   ├── test/images/             # Тестовые фото (10 шт)
│   └── test/labels/             # YOLO-разметка
│   ├── model.pt                 ← Веса YOLO (копировать после обучения)
│   └── runs/                    ← Результаты обучения (игнорируется git)
├── mahjong_layout/              # Пакет классификации и зонирования
│   ├── pipeline.py              # classify_layout() — основной пайплайн
│   ├── classify/                # Классификатор тайлов
│   └── types.py                 # TileBox, LayoutResult и т.д.
├── scripts/
│   ├── yolo_pipeline.py         # YOLO → classify_layout → overlay
│   └── viz_classify.py          # Визуализация на valid-выборке
├── tests/                       # Синтетические тесты + боевые фото
├── out/                         # Результаты прогонов (игнорируется git)
├── requirements.txt             # Зависимости для mahjong_layout
├── pyproject.toml               # pip install -e .
├── README.md                    # Документация mahjong_layout
└── README_YOLO.md               ← Этот файл
```

**Важно:** YOLO детектит все тайлы как один класс (86 классов в Roboflow-датасете,
но `yolo_pipeline.py` использует только координаты, не class_id).
Классификацию по мастям (wan/pin/tiao/honor + значение) делает `classify_tile` на кропе.
