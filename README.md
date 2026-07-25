# mahjong-layout

Пространственная кластеризация детекций тайлов маджонга в зоны **рука / сброс / стена**.
Модуль принимает готовые детекции (bbox'ы от YOLO или любого другого детектора) и
группирует их в кластеры, после чего эвристически помечает роли. Классификацию
**типов тайлов** (dot/tiao/wan) не делает — только геометрия.

## Идея

- **Кластеризация** — scale-aware DBSCAN на numpy. Радиус соседства `eps`
  привязан к медианному размеру тайла (`eps = eps_k * median_tile_size`), поэтому
  один и тот же порог работает и на крупном плане, и на широком снимке стола.
- **Роли** — по упрощению из брифа: *нижняя часть кадра = рука*. Hand-кластер
  выбирается по вертикальной позиции, уточняется ориентацией и регулярностью
  ряда. Остальное делится на wall (стоячие ряды) и discard (россыпь/стопки).
- **Без тяжёлых зависимостей**: только `numpy` + `pillow`. Никаких `cv2`/
  `sklearn`/`ultralytics` — кластеризация своя (~80 строк), viz на PIL.
- **YOLO-инференс снаружи**: модуль — потребитель детекций. Когда появится
  обученная модель, она подключается без изменений ядра.

## Установка

Зависимости — только `numpy` и `pillow` (оба уже есть в окружении).

```bash
pip install -e .            # из директории проекта
```

На системах с PEP 668 (externally-managed environment) `pip install` без флагов
заблокирован. Варианты:

```bash
pip install -e . --break-system-packages   # если доверяете окружению
# или
pipx install -e .                           # изолированно через pipx
# или запускать без установки:
python -m mahjong_layout.cli <args>         # CLI
python -m pytest                            # тесты
```

Для тестов: `pip install pytest`.

## Использование как библиотека

```python
from mahjong_layout import cluster_layout, LayoutParams, TileBox

# Допустим, boxes — выход вашей YOLO: список TileBox/кортежей/словарей.
# Координаты нормализованы (как в YOLO): cx, cy, w, h в [0, 1].
result = cluster_layout(boxes, image_size=(640, 640))

print(result.summary())          # "hand=13 discard=1c(5) wall=1c(4) other=0"
print(result.hand.n_tiles)       # 13
print([(c.label, c.n_tiles) for c in result.discards])  # [("discard", 5)]

# Свои пороги:
params = LayoutParams(hand_y_min=0.55, eps_k=3.0, discard_min_tiles=2)
result = cluster_layout(boxes, params=params)
```

Форматы входа (через `mahjong_layout.io_readers`):
- `TileBox(cx, cy, w, h, class_id=None)` — основной;
- кортеж `(cx, cy, w, h)` или `(class_id, cx, cy, w, h)`;
- словарь `{"cx":..,"cy":..,"w":..,"h":..,"class_id":..}`;
- YOLO `.txt`: `class cx cy w h`;
- JSON: массив объектов того же формата.

## CLI

```bash
# Сводка по каждому фото (читает .txt label-файлы YOLO):
mahjong-layout Mahjong_YOLO.v2i.yolo26/valid/labels

# Полный прогон: JSON + отрисовка поверх фото:
mahjong-layout valid/labels \
    --images-dir ../images \
    --out out --json --viz
```

Вывод `--json` (`out/layout.json`):

```json
{
  "p1": {
    "summary": "hand=13 discard=1c(5) wall=0 other=0",
    "hand": { "role": "hand", "n_tiles": 13, "centroid": [0.46, 0.85], "regularity": 0.99, ... },
    "discards": [...],
    "walls": [...],
    "others": [...]
  }
}
```

Параметры CLI зеркалят поля `LayoutParams`: `--hand-y-min`, `--eps-k`,
`--min-samples`, `--hand-max-tiles`, `--discard-min-tiles`, `--wall-aspect`.

## Параметры и эвристики (`LayoutParams`)

| Поле               | По умолч. | Что делает                                                            |
|--------------------|-----------|-----------------------------------------------------------------------|
| `eps_k`            | 2.5       | `eps = eps_k * median_tile_size` — радиус соседства в DBSCAN.         |
| `min_samples`      | 2         | Минимум соседей для core-точки; одиночки → `other`.                   |
| `hand_y_min`       | 0.60      | Нижняя зона кадра = рука. Центроид hand-кандидата должен быть ниже.   |
| `hand_max_tiles`   | 18        | Рука = 13–14 тайлов + запас наmelds; больше → не hand.                |
| `hand_max_rows`    | 3         | Рука занимает мало рядов; стопки/стены отсекаются.                    |
| `wall_aspect`      | 1.5       | `h/w > wall_aspect` у стоячего кластера → `wall`.                     |
| `discard_min_tiles`| 2         | Кластеры от этого размера (после hand/wall) → `discard`; меньше → `other`. |

Все пороги вынесены в один дата-класс — тюнить можно без правки логики.

## Классификация тайлов (`mahjong_layout.classify`)

Вторая часть модуля — определение **масти и номинала** отдельного тайла по
кропу. Архитектура двухступенчатая:

1. **Suit router** (`determine_suit`) — определяет масть по содержимому:
   - **wan** — через OCR-маркер 萟/万 (RapidOCR). Это единственный надёжный
     сигнал для wan, т.к. штрихи иероглифов геометрически неотличимы от палочек
     бамбука. Идёт первым, когда OCR доступен.
   - **pin (точки)** — много круглых связных компонентов.
   - **tiao (бамбуки)** — много тонких вертикальных компонентов; **tiao1 = птица**
     (один большой blob) — особый случай.
   - **honor/unknown** — иначе.
2. **Decoder по масти**:
   - `wan` → RapidOCR на иероглифе-числе + мапа 一→1 … 九→9 (`CN_NUMERAL_TO_INT`).
   - `pin`/`tiao` → `cv2.connectedComponents` + модальная фильтрация по площади
     (точки/палочки одного тайла одинакового размера → считаем кластер
     одинаковой площади, а не все компоненты подряд).
   - `honor` → stub (нет данных по ветрам/драгонам); белый дракон помечается
     как `white_dragon_candidate`.

```python
from PIL import Image
from mahjong_layout.classify import classify_tile, crop_tile
from mahjong_layout import TileBox

crop = crop_tile("photo.jpg", TileBox(0.5, 0.85, 0.06, 0.08))
res = classify_tile(crop)
print(res.label, res.value, res.confidence, res.method)
# напр. "pin5 5 0.82 count_components"  или  "wan3 3 0.71 ocr"
```

### Установка OCR-зависимостей

```bash
pip install -e ".[ocr]"   # rapidocr-onnxruntime + opencv-python + numpy2/scipy1.13
```

Важно: версия `numpy`/`scipy`/`opencv` должны быть согласованы (opencv 5 требует
numpy>=2, scipy 1.11 требует numpy<2 → конфликт). Extra `[ocr]` уже пинит
совместимую тройку `numpy>=2,<3` + `scipy>=1.13`. **OCR — optional dependency**:
без неё классификатор деградирует до UNKNOWN (не падает), а модуль кластеризации
работает как прежде.

### Устойчивость к качеству: каскад методов

CV-ядро построено как **каскад от сильного сигнала к слабому**, чтобы работать
и на качественных фото, и на плохих:

1. **Цветовая сегментация** (`color_mask`) — на цветных фото точки/палочки
   цветные на белом лице тайла, это самый надёжный сигнал. На grayscale-фото
   (часть датасета обесцвечена) путь пропускается автоматически.
2. **Авто-детект лица тайла** (`detect_face`) — находит самый крупный светлый
   прямоугольный регион, отсекая рамку тайла, которая иначе доминирует в
   connected-components. Fallback на фиксированный inset, если не получилось.
3. **Adaptive threshold** — luminance-бинаризация, когда цвет недоступен.
4. **Модальная фильтрация по площади** — точки/палочки одного тайла одного
   размера, поэтому считаем кластер одинаковых по площади компонентов, а не
   все подряд (отсев шума и рамки).

Параметры — в `ClassifyParams`: `color_tile`, `color_saturation_min`,
`detect_face`, `use_modal_area_filter`.

### Прогон по своим фото

```bash
python scripts/classify_photo.py --tile photo.jpg --box "0.5 0.85 0.06 0.08" --debug
# -> 421018208..._classified.png с подписью номинала
```

`--box` — нормализованные координаты тайла (cx cy w h). `--debug` сохраняет
аннотированный кроп в `--out`. Флаги `--no-color` / `--no-face-detect` помогают
калибровать, какой метод лучше работает на ваших фото.

### Честное состояние

Smoke-тест на реальных кропах датасета (`Mahjong_YOLO.v2i.yolo26`) показал:

- **Синтетика (вкл. цветовые тайлы)**: 92/92 теста зелёные. Цветовой путь
  точно считает цветные точки (pin3→3, pin5→5, pin9→9 на синтетике).
- **Реальные кропы датасета**: точность **низкая**. Доказанные причины:
  - Кропы крошечные (18–40 px). **Проверено**: апскейл того же кропа (x2/x4/x8)
    не восстанавливает числовой иероглиф wan — информация теряется при съёмке.
  - Часть фото **обесцвечена** (grayscale-as-RGB) → цветовой путь недоступен.
- Что **работает на реальных**: определение масти pin (точки), детекция wan
  через OCR-маркер 萟 (когда OCR его видит).

Вывод: модуль **готов к качественным фото** (≥150px/тайл, цветные), но текущий
датасет — не тот вход для высоких результатов. Нужны фото лучшего качества.

## Структура

```
mahjong_layout/
├── types.py        # TileBox, Cluster, LayoutParams, LayoutResult
├── clustering.py   # scale-aware DBSCAN + дескрипторы кластеров
├── heuristics.py   # пометка hand/discard/wall/other
├── pipeline.py     # cluster_layout() — высокоуровневый API
├── io_readers.py   # YOLO .txt / JSON / raw → list[TileBox]
├── viz.py          # отрисовка кластеров поверх фото (PIL)
├── cli.py          # entrypoint mahjong-layout
├── crop.py         # crop_tile(image, TileBox) → PIL.Image
└── classify/       # классификация тайлов (optional OCR)
    ├── types.py        # Suit, TileClassification, ClassifyParams
    ├── constants.py    # масти, мапа иероглифов-чисел, маркеры 萟
    ├── preprocess.py   # upscale + inset + adaptiveThreshold
    ├── router.py       # determine_suit() — маршрутизация по масти
    ├── ocr_engine.py   # обёртка RapidOCR (lazy, изоляция сбоев)
    ├── wan_decoder.py  # OCR иероглифа-числа → 1..9
    ├── count_decoder.py# connectedComponents для pin/tiao → N
    ├── honor.py        # stub под ветра/драконы
    └── classifier.py   # classify_tile() — высокоуровневый API
scripts/
└── classify_photo.py  # прогон классификации по своим фото (--tile/--box)
tests/              # синтетические тесты (датасет не используется)
```

## Тесты

```bash
pytest -q
```

Тесты синтетические (датасет намеренно не трогается): покрывают кластеризацию
(scale-aware eps, разделение рядов, регулярность), эвристики ролей и
сквозной пайплайн + IO-ридеры.

## Что дальше

- **YOLO-инференс**: обёртка над `ultralytics` → `list[TileBox]` → `cluster_layout`
  → кропы → `classify_tile`. Это даст полный пайплайн «фото → рука/сброс + типы
  тайлов» и, что важно, более крупные/чистые кропы для классификации.
- **CV-тюн под реальные тайлы**: цветовая сегментация точек (пины цветные),
  авто-определение границы тайла вместо фиксированного inset, адаптация
  бинаризации под освещение. Текущий inset + modal-filter делает масть pin
  стабильной; tiao/wan на мелких кропах датасета требуют этих доработок.
- **Honor-декодер**: мапа символов 東南西北中發白 + белый дракон как corner-case
  (заложен stub + `white_dragon_candidate`).
- **Сквозная CLI**: единая команда «фото → раскладка + классификация тайлов».
- Когда появится классификация типов тайлов, добавить слой «содержимое кластера»
  поверх существующих дескрипторов (роли не зависят от типов тайлов).
