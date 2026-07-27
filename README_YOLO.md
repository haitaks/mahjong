# Mahjong YOLO — обучение детекции тайлов

## Что нужно сделать

На другой машине с GPU (или мощным CPU) обучить YOLOv8n детектить тайлы маджонга.
Готовые веса скопировать обратно на эту машину.

---

## Шаг 1 — скопировать данные

На этой машине:

```bash
scp /home/rosalvak/projects/mahjong/yolo_dataset.tar.gz user@target-machine:~/mahjong_yolo/
```

На целевой машине:

```bash
cd ~/mahjong_yolo
tar xzf yolo_dataset.tar.gz
# появится папка yolo/ с train/valid/test/data.yaml
```

---

## Шаг 2 — установить зависимости

```bash
pip install ultralytics
# или:
# conda install -c conda-forge ultralytics
```

---

## Шаг 3 — обучить

```bash
cd ~/mahjong_yolo/yolo
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); \
  model.train(data='data.yaml', epochs=100, imgsz=640, batch=8, device='cuda')"
```

Параметры:
- `epochs=100` — норм для детекции одного класса (все тайлы)
- `batch=8` — подогнать под VRAM (если 4GB → batch=4, если 6GB → batch=8)
- `device='cuda'` — для GPU. На CPU уберите или `device='cpu'`
- `imgsz=640` — как в датасете

После обучения веса будут в `runs/detect/train/weights/best.pt`.

---

## Шаг 4 — скопировать веса обратно

```bash
scp ~/mahjong_yolo/yolo/runs/detect/train/weights/best.pt \
  user@this-machine:/home/rosalvak/projects/mahjong/yolo/model.pt
```

---

## Шаг 5 — запустить полный пайплайн на этой машине

```bash
cd /home/rosalvak/projects/mahjong
python scripts/yolo_pipeline.py
```

Результаты: `out/yolo_pipeline/*_yolo_pipeline.jpg` — фото с цветными рамками и подписями.

---

## Структура проекта (для справки)

```
mahjong/
├── yolo/
│   ├── model.pt              ← веса YOLO (скопировать после обучения)
│   ├── data.yaml              ← конфиг датасета
│   ├── train/images/          ← тренировочные фото
│   ├── train/labels/          ← YOLO-разметка
│   ├── valid/images/          ← валидационные фото
│   └── valid/labels/          ← YOLO-разметка
├── tests/
│   ├── photo_*.jpg            ← боевые фото для теста
├── mahjong_layout/
│   ├── pipeline.py            ← classify_layout (классификация + кластеризация)
│   ├── classify/              ← классификатор тайлов
│   └── types.py               ← TileBox, LayoutResult, etc.
├── scripts/
│   ├── yolo_pipeline.py       ← YOLO → classify_layout → overlay
│   └── ...
└── requirements.txt           ← зависимости для mahjong_layout
```

**Важно:** YOLO детектит **все** тайлы как один класс (class_id без разницы).
Классификацию (wan/pin/tiao/honor + значение) делает `classify_tile` на кропе.
