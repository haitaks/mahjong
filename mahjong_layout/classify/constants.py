"""Constants for tile classification: suit names, value ranges, OCR maps."""

from __future__ import annotations

# --- Suit names (string values match Suit enum) -------------------------
SUIT_WAN = "wan"
SUIT_PIN = "pin"
SUIT_TIAO = "tiao"
SUIT_HONOR = "honor"
SUIT_UNKNOWN = "unknown"

# Valid numeric value range for suited tiles (1..9).
MIN_VALUE = 1
MAX_VALUE = 9

# --- OCR: Chinese numeral -> int ----------------------------------------
# wan tiles draw the number as a Chinese numeral (一/二/.../九). We also accept
# the formal "bank" variants (壹貳...) and Western digits as a fallback when
# the OCR model hallucinates digits instead of characters.
CN_NUMERAL_TO_INT = {
    # simple
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9,
    # formal / bank (大写)
    "壹": 1, "貳": 2, "贰": 2, "叁": 3, "參": 3,
    "肆": 4, "伍": 5, "陸": 6, "陆": 6,
    "柒": 7, "捌": 8, "玖": 9,
    # western digit fallback
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    "6": 6, "7": 7, "8": 8, "9": 9,
    # zero is not a valid suited value but map it for completeness
    "零": 0, "〇": 0, "0": 0,
}

# Markers identifying the wan suit: the 萬 / 万 character sits at the bottom of
# the tile. Presence strongly implies the wan suit.
WAN_MARKERS = {"萬", "万"}

# Honor characters (winds + dragons). Each honor class maps to the set of
# characters that may appear on its face. Used by the honor decoder.
HONOR_CHARS = {
    "east": {"东", "東"},
    "south": {"南"},
    "west": {"西"},
    "north": {"北"},
    "red_dragon": {"中"},
    "green_dragon": {"发", "發"},
}

# Reverse lookup: character -> honor label.
HONOR_CHAR_TO_LABEL = {ch: label for label, chars in HONOR_CHARS.items() for ch in chars}

# White dragon has no character — it's a blank/framed tile, detected by very
# low foreground density, not by OCR.
WHITE_DRAGON_LABEL = "white_dragon"

# Color hints for the honor fallback (when OCR can't read the character).
# red_dragon 中 is red; green_dragon 發 is green. Winds are typically black.
HONOR_COLOR_HINTS = {
    "red_dragon": "red",
    "green_dragon": "green",
}


def make_label(suit: str, value: int | None) -> str:
    """Build a canonical label like 'wan5', 'pin9', 'tiao1'."""
    if value is None:
        return suit
    return f"{suit}{value}"
