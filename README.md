# mahjong-layout

Классификация и пространственная зонация тайлов маджонга по фото стола.

Пайплайн: детекция → классификация каждого тайла (кроп из фото) → поиск руки по самому нижнему осмысленному тайлу → определение стены (пустые тайлы далеко от осмысленных) → сброс (остальные).

Кластеризация не используется — только классификация + простые геометрические эвристики.

## Установка

Зависимости: `numpy` + `pillow`. Для классификации опционально `rapidocr-onnxruntime` + `opencv-python`.

```bash
pip install -e .                              # базовая установка (без OCR)
pip install -e ".[ocr]"                       # с OCR-зависимостями
```

На системах с PEP 668 (externally-managed environment):

```bash
pip install -e . --break-system-packages
# или
pipx install -e .
# или запуск без установки:
python -m mahjong_layout.cli <args>
```

## Использование как библиотека

```python
from PIL import Image
from mahjong_layout import classify_layout, LayoutParams, TileBox

# boxes — список детекций от YOLO/любого детектора
# Координаты нормализованы (cx, cy, w, h в [0, 1]).
image = Image.open("photo.jpg")
boxes = [TileBox(0.5, 0.85, 0.06, 0.08)]

result = classify_layout(boxes, image)

print(result.summary())        # "hand=13 discard=5 wall=4 unknown=0"
print(len(result.hand))        # 13
print(len(result.discard))     # 5
print(len(result.wall))        # 4

# Доступ к классифицированным тайлам:
for ct in result.hand:
    print(ct.label, ct.box.cx, ct.box.cy)  # "wan5" / "pin3" / "unknown"
```

### Форматы входа

- `TileBox(cx, cy, w, h, class_id=None)` — основной формат;
- кортеж `(cx, cy, w, h)` или `(class_id, cx, cy, w, h)`;
- словарь `{ "cx": ..., "cy": ..., "w": ..., "h": ..., "class_id": ... }`;
- YOLO `.txt`: `class cx cy w h`;
- JSON: массив объектов того же формата (конвертация через `mahjong_layout.io_readers`).

### Свои пороги

```python
params = LayoutParams(hand_eps_k=3.0, wall_min_tiles=5)
result = classify_layout(boxes, image, params=params)
```

## CLI

```bash
# Сводка по каждому фото (читает .txt label-файлы YOLO и сами фото):
mahjong-layout valid/labels --images-dir ../images

# Полный прогон с JSON-выводом:
mahjong-layout valid/labels --images-dir ../images --out out --json
```

Параметры CLI зеркалят поля `LayoutParams`:
`--hand-eps-k`, `--hand-max-tiles`, `--eps-k`, `--min-samples`,
`--wall-neighbor-eps`, `--wall-min-tiles`.

## Параметры и эвристики (`LayoutParams`)

| Поле               | Умолч. | Что делает                                                   |
|-------------------|--------|--------------------------------------------------------------|
| `classify_params` | `None` | Словарь-переопределение параметров классификации (`ClassifyParams`). |
| `hand_eps_k`      | 4.0    | Радиус поиска руки: `eps = hand_eps_k * median_tile_size` от самого нижнего (max cy) осмысленного тайла. |
| `hand_max_tiles`  | 14     | Максимум тайлов в руке (13 + 1). Лишние → discard.          |
| `eps_k`           | 1.5    | (Не используется в текущем пайплайне, зарезервирован для будущей кластеризации.) |
| `min_samples`     | 2      | (Не используется, зарезервирован.)                          |
| `wall_neighbor_eps` | 0.08 | Пустой тайл считается «рядом» с осмысленным, если расстояние < этого порога (норм. коорд.). Иначе — кандидат стену. |
| `wall_min_tiles`  | 3      | Группа пустых тайлов должна быть ≥ этого размера, чтобы считаться стеной. Меньшие группы → unknown. |

## Классификация тайлов (`mahjong_layout.classify`)

Двухступенчатая архитектура:

1. **Suit router** (`determine_suit`) — определяет масть:
   - **wan** — через OCR-маркер 萟/万 (RapidOCR).
   - **pin (точки)** — много круглых связных компонентов.
   - **tiao (бамбуки)** — много тонких вертикальных компонентов; tiao1 = птица (один большой blob).
   - **honor** — OCR ветров/драконов + цветовой fallback (красный → red_dragon, зелёный → green_dragon).
   - **unknown** — пустой тайл (стена/неразборчиво).
2. **Decoder по масти**:
   - `wan` → RapidOCR на иероглифе-числе + мапа 一→1 … 九→9.
   - `pin`/`tiao` → cv2.connectedComponents + модальная фильтрация по площади.
   - `honor` → OCR символов 東南西北中發白 + пустой тайл → white_dragon.

```python
from PIL import Image
from mahjong_layout import classify_tile, crop_tile, TileBox

crop = crop_tile("photo.jpg", TileBox(0.5, 0.85, 0.06, 0.08))
res = classify_tile(crop)
print(res.label, res.value, res.confidence, res.method)
# "pin5 5 0.82 count_components" или "wan3 3 0.71 ocr"
```

## Структура модуля

```
mahjong_layout/
├── types.py             # TileBox, ClassifiedTile, LayoutParams, LayoutResult
├── pipeline.py          # classify_layout() — основной пайплайн
├── io_readers.py        # YOLO .txt / JSON / raw → list[TileBox]
├── crop.py              # crop_tile(image, TileBox) → PIL.Image
├── cli.py               # entrypoint mahjong-layout
├── classify/            # классификация тайлов (optional OCR)
│   ├── types.py         # Suit, TileClassification, ClassifyParams
│   ├── constants.py     # масти, мапа иероглифов, маркеры 萟
│   ├── preprocess.py    # upscale + inset + adaptiveThreshold
│   ├── router.py        # determine_suit() — маршрутизация по масти
│   ├── ocr_engine.py    # обёртка RapidOCR (lazy, изоляция сбоев)
│   ├── wan_decoder.py   # OCR иероглифа-числа → 1..9
│   ├── count_decoder.py # connectedComponents для pin/tiao → N
│   ├── honor.py         # декодер ветров/драконов + white_dragon
│   └── classifier.py    # classify_tile() — высокоуровневый API
scripts/
├── classify_photo.py    # прогон классификации по своим фото
└── quick_test.py        # быстрый тест пайплайна с локальными фото
tests/                   # синтетические тесты
```

## Тесты

```bash
python -m pytest -q
```

Тесты синтетические (датасет не трогается): покрывают классификацию, пайплайн, IO-ридеры, декодеры, обработку honor-тайлов.

## Что дальше

- **YOLO-инференс**: обёртка над `ultralytics` → `list[TileBox]` → `classify_layout` → полный пайплайн фото → раскладка + типы тайлов.
- **Сквозная CLI**: единая команда «фото → раскладка + классификация».
- **CV-тюн под реальные тайлы**: цветовая сегментация точек, авто-определение границы тайла, адаптация бинаризации под освещение.
