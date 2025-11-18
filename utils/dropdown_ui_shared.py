from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import random
import cv2
import numpy as np

# Shared dropdown UI generation and validation utilities

FONT_FACES = {
    "SIMPLEX": cv2.FONT_HERSHEY_SIMPLEX,
    "PLAIN": cv2.FONT_HERSHEY_PLAIN,
    "DUPLEX": cv2.FONT_HERSHEY_DUPLEX,
    "COMPLEX": cv2.FONT_HERSHEY_COMPLEX,
    "TRIPLEX": cv2.FONT_HERSHEY_TRIPLEX,
}

SAMPLE_WORDS = [
    "Apple","Banana","Cherry","Dragon","Elephant","Forest","Galaxy","Horizon","Island","Jungle",
    "Koala","Lemon","Mountain","Neptune","Ocean","Phoenix","Quantum","River","Saturn","Thunder",
    "Universe","Volcano","Whisper","Xenon","Yellow","Zebra","Alpha","Beta","Gamma","Delta","Sigma"
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def clip_to_canvas(img: np.ndarray, x: int, y: int, w: int, h: int) -> Tuple[int,int,int,int]:
    H, W = img.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def rand_color() -> Tuple[int,int,int]:
    return (random.randint(40,255), random.randint(40,255), random.randint(40,255))


def light_color() -> Tuple[int,int,int]:
    return (random.randint(220,250), random.randint(220,250), random.randint(220,250))


def text_contrast(bg: Tuple[int,int,int]) -> Tuple[int,int,int]:
    return (30,30,30) if sum(bg)/3 > 128 else (240,240,240)


def draw_gradient(img: np.ndarray, x: int, y: int, w: int, h: int,
                  c1: Tuple[int,int,int], c2: Tuple[int,int,int], vertical: bool=True) -> None:
    x, y, w, h = clip_to_canvas(img, x, y, w, h)
    if w <= 0 or h <= 0:
        return
    if vertical:
        alpha = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        c1v = np.array(c1, np.float32)[None, :]
        c2v = np.array(c2, np.float32)[None, :]
        line = (c1v * (1 - alpha) + c2v * alpha)
        band = np.repeat(line[:, None, :], w, axis=1).astype(np.uint8)
    else:
        alpha = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
        c1v = np.array(c1, np.float32)[:, None]
        c2v = np.array(c2, np.float32)[:, None]
        line = (c1v * (1 - alpha) + c2v * alpha).T
        band = np.repeat(line[None, :, :], h, axis=0).astype(np.uint8)
    img[y:y+h, x:x+w] = band


def draw_shadow(img: np.ndarray, x: int, y: int, w: int, h: int, offset: int=4, alpha: float=0.25) -> None:
    sx, sy = x + offset, y + offset
    sx, sy, w, h = clip_to_canvas(img, sx, sy, w, h)
    if w <= 0 or h <= 0:
        return
    region = img[sy:sy+h, sx:sx+w].astype(np.float32)
    img[sy:sy+h, sx:sx+w] = (region * (1 - alpha)).astype(np.uint8)


class Style:
    def __init__(self, idx: int) -> None:
        self.id = idx
        self.h_bg1 = rand_color() if idx % 4 == 0 else light_color()
        self.h_bg2 = rand_color() if idx % 5 == 0 else self.h_bg1
        self.h_text = text_contrast(self.h_bg1)
        self.h_radius = random.randint(0, 10)
        self.h_border = random.randint(0, 3)
        self.h_shadow = random.random() < 0.5
        self.p_bg1 = light_color(); self.p_bg2 = rand_color()
        self.p_border = random.randint(0, 2)
        self.p_radius = random.randint(0, 8)
        self.p_shadow = random.random() < 0.6
        self.h_font = random.choice(list(FONT_FACES.keys()))
        self.h_scale = random.uniform(0.45, 0.75)
        self.h_thick = random.randint(1, 2)
        self.o_font = random.choice(list(FONT_FACES.keys()))
        self.o_scale = random.uniform(0.35, 0.6)
        self.o_thick = random.randint(1, 2)
        self.caret_right = random.random() < 0.85
        self.scroll_w = random.randint(4, 7)


class State:
    def __init__(self, open_prob: float) -> None:
        self.is_open = (random.random() < open_prob)
        self.header_text = f"Select: {random.choice(SAMPLE_WORDS)} {random.randint(1,999)}"
        if self.is_open:
            self.total = random.randint(8, 30)
            self.visible = random.randint(3, min(8, self.total))
            self.scroll = random.randint(0, max(0, self.total - self.visible))
            self.row_h = random.randint(24, 34)
            self.texts = [f"{random.choice(SAMPLE_WORDS)} {random.randint(1,999)}" for _ in range(self.total)]
            self.hover = None; self.selected = None
            if random.random() < 0.3:
                self.hover = self.scroll + random.randint(0, self.visible - 1)
            if random.random() < 0.3:
                self.selected = random.randint(0, self.total - 1)
        else:
            self.total = 0; self.visible = 0; self.scroll = 0; self.row_h = 0; self.texts = []
            self.hover = None; self.selected = None


def draw_dropdown(canvas: np.ndarray, style: Style, state: State,
                  x: int, y: int, w: int, h: int) -> Dict[str, Any]:
    # Header
    if style.h_shadow:
        draw_shadow(canvas, x, y, w, h, 3, 0.2)
    draw_gradient(canvas, x, y, w, h, style.h_bg1, style.h_bg2, True)
    if style.h_border > 0:
        col = (max(0, style.h_bg1[0]-60), max(0, style.h_bg1[1]-60), max(0, style.h_bg1[2]-60))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), col, style.h_border)
    font = FONT_FACES[style.h_font]
    ts = cv2.getTextSize(state.header_text, font, style.h_scale, style.h_thick)[0]
    cv2.putText(canvas, state.header_text, (x + 8, y + h // 2 + ts[1] // 2), font, style.h_scale, style.h_text, style.h_thick, cv2.LINE_AA)
    if style.caret_right:
        cx = x + w - 14; cy = y + h // 2
        cv2.polylines(canvas, [np.array([[cx - 5, cy - 4], [cx + 2, cy], [cx - 5, cy + 4]], dtype=np.int32)], False, (30, 30, 30), 2, cv2.LINE_AA)

    meta: Dict[str, Any] = {
        "styleId": style.id,
        "state": ("open" if state.is_open else "closed"),
        "headerBox": {"x": x, "y": y, "w": w, "h": h},
        "panelBox": None,
        "options": [],
    }

    scroll_meta: Dict[str, Any] = {"total": state.total, "visible": state.visible, "scrollIndex": state.scroll, "hasScrollbar": False, "scrollbar": {"track": None, "thumb": None}}
    options: List[Dict[str, Any]] = []
    panel_box = None

    if state.is_open:
        px, py = x, y + h; pw, ph = w, state.visible * state.row_h
        panel_box = {"x": px, "y": py, "w": pw, "h": ph}
        if style.p_shadow:
            draw_shadow(canvas, px, py, pw, ph, 4, 0.25)
        draw_gradient(canvas, px, py, pw, ph, style.p_bg1, style.p_bg2, True)
        if style.p_border > 0:
            col = (max(0, style.p_bg1[0]-40), max(0, style.p_bg1[1]-40), max(0, style.p_bg1[2]-40))
            cv2.rectangle(canvas, (px, py), (px + pw, py + ph), col, style.p_border)

        # Options
        for gi in range(state.total):
            if state.scroll <= gi < state.scroll + state.visible:
                vi = gi - state.scroll; oy = py + vi * state.row_h
                row_box = {"x": px, "y": oy, "w": pw, "h": state.row_h}
                text = state.texts[gi]
                ofont = FONT_FACES[style.o_font]
                ts2 = cv2.getTextSize(text, ofont, style.o_scale, style.o_thick)[0]
                cv2.putText(canvas, text, (px + 8, oy + state.row_h // 2 + ts2[1] // 2), ofont, style.o_scale, (30, 30, 30), style.o_thick, cv2.LINE_AA)
                options.append({
                    "index": gi,
                    "visibleIndex": vi,
                    "text": text,
                    "box": row_box,
                    "center": {"x": px + pw // 2, "y": oy + state.row_h // 2},
                    "isHover": (gi == (state.hover or -1)),
                    "isSelected": (gi == (state.selected or -1)),
                })
            else:
                options.append({
                    "index": gi,
                    "visibleIndex": None,
                    "text": (state.texts[gi] if state.texts else ""),
                    "box": None,
                    "center": None,
                    "isHover": False,
                    "isSelected": (gi == (state.selected or -1)),
                })

        # Scrollbar
        if state.total > state.visible:
            scroll_meta["hasScrollbar"] = True
            tw = max(4, style.scroll_w)
            tr_x, tr_y, tr_w, tr_h = px + pw - tw - 2, py + 2, tw, ph - 4
            cv2.rectangle(canvas, (tr_x, tr_y), (tr_x + tr_w, tr_y + tr_h), (200, 200, 200), -1)
            thumb_h = max(8, int(tr_h * state.visible / max(1, state.total)))
            max_scroll = max(0, state.total - state.visible)
            off = int((tr_h - thumb_h) * (state.scroll / max(1, max_scroll))) if max_scroll > 0 else 0
            th_y = tr_y + off
            cv2.rectangle(canvas, (tr_x, th_y), (tr_x + tr_w, th_y + thumb_h), (120, 120, 120), -1)
            scroll_meta["scrollbar"] = {
                "track": {"x": tr_x, "y": tr_y, "w": tr_w, "h": tr_h},
                "thumb": {"x": tr_x, "y": th_y, "w": tr_w, "h": thumb_h},
            }

    meta.update({
        "panelBox": panel_box,
        "options": options,
        "scroll": scroll_meta,
        "theme": {
            "bg": ("gradient" if True else "solid"),
            "panel": "gradient",
            "radii": {"header": getattr(style, "h_radius", 0), "panel": getattr(style, "p_radius", 0)},
            "borders": {"header": getattr(style, "h_border", 0), "panel": getattr(style, "p_border", 0)},
            "shadows": {"header": getattr(style, "h_shadow", False), "panel": getattr(style, "p_shadow", False)},
        },
        "fonts": {
            "header": {"face": "SIMPLEX", "scale": style.h_scale, "thick": style.h_thick},
            "option": {"face": "PLAIN", "scale": style.o_scale, "thick": style.o_thick},
        }
    })
    return meta


def render_grid(canvas_w: int = 1600, canvas_h: int = 1000, cols: int = 5, rows: int = 8,
                seed: Optional[int] = 42, expanded_prob: float = 0.6,
                limit: int = 40) -> Tuple[np.ndarray, Dict[str, Any]]:
    if seed is not None:
        set_seed(seed)
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 245
    margin = 20
    cell_w = (canvas_w - 2 * margin) // max(1, cols)
    cell_h = (canvas_h - 2 * margin) // max(1, rows)
    metas: List[Dict[str, Any]] = []
    count = 0
    for r in range(rows):
        for c in range(cols):
            if count >= min(cols * rows, limit):
                break
            base_x = margin + c * cell_w
            base_y = margin + r * cell_h
            x = max(5, min(canvas_w - 200, base_x + random.randint(-10, 10)))
            y = max(5, min(canvas_h - 60, base_y + random.randint(-10, 10)))
            w = random.randint(int(cell_w * 0.6), int(cell_w * 0.85))
            h = random.randint(28, 42)
            style = Style(count)
            state = State(expanded_prob)
            meta = draw_dropdown(canvas, style, state, x, y, w, h)
            metas.append(meta)
            count += 1
    payload = {
        "canvas_w": canvas_w,
        "canvas_h": canvas_h,
        "cols": cols,
        "rows": rows,
        "seed": seed,
        "expanded_prob": expanded_prob,
        "dropdowns": metas,
    }
    return canvas, payload


def render_transition_sequence(canvas_w: int = 600, canvas_h: int = 300,
                               seed: Optional[int] = 123,
                               total: int = 15, visible: int = 5,
                               row_h: int = 28) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """
    Create an idealized dropdown interaction sequence using the same visuals as validation:
      1) closed header
      2) open at top
      3..N) progressively scroll down to bottom
      last) click/select bottom-visible option

    Returns a list of frames and corresponding single-dropdown meta payloads.
    """
    if seed is not None:
        set_seed(seed)
    # Fixed placement
    x, y, w, h = 40, 40, 360, 36
    frames: List[np.ndarray] = []
    metas: List[Dict[str, Any]] = []

    # Build a fixed style/state baseline
    style = Style(0)
    state = State(1.0)  # force open
    state.total = max(visible + 1, total)
    state.visible = max(3, visible)
    state.row_h = row_h
    state.scroll = 0

    # 1) Closed frame
    closed_canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 245
    st_closed = State(0.0)
    st_closed.header_text = state.header_text
    meta1 = draw_dropdown(closed_canvas, style, st_closed, x, y, w, h)
    frames.append(closed_canvas)
    metas.append({"canvas_w": canvas_w, "canvas_h": canvas_h, "dropdowns": [meta1]})

    # 2) Open at top
    open_canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 245
    state.scroll = 0
    meta2 = draw_dropdown(open_canvas, style, state, x, y, w, h)
    frames.append(open_canvas)
    metas.append({"canvas_w": canvas_w, "canvas_h": canvas_h, "dropdowns": [meta2]})

    # 3..k) Scroll down steps
    max_scroll = max(0, state.total - state.visible)
    for s in range(1, max_scroll + 1):
        sc_canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 245
        state.scroll = s
        meta_s = draw_dropdown(sc_canvas, style, state, x, y, w, h)
        frames.append(sc_canvas)
        metas.append({"canvas_w": canvas_w, "canvas_h": canvas_h, "dropdowns": [meta_s]})

    # Final) Select bottom-visible
    sel_canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 245
    state.scroll = max_scroll
    # mark selected as currently last visible option
    state.selected = min(state.total - 1, state.scroll + state.visible - 1)
    meta_last = draw_dropdown(sel_canvas, style, state, x, y, w, h)
    frames.append(sel_canvas)
    metas.append({"canvas_w": canvas_w, "canvas_h": canvas_h, "dropdowns": [meta_last]})

    return frames, metas


def validate_meta(meta_path: Path) -> bool:
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    errs: List[str] = []
    for i, dd in enumerate(data.get("dropdowns", [])):
        if dd.get("state") == "open":
            if dd.get("panelBox") is None:
                errs.append(f"#{i}: panelBox None for open")
            for opt in dd.get("options", []):
                if opt.get("visibleIndex") is not None:
                    b, c = opt.get("box"), opt.get("center")
                    if b is None or c is None:
                        errs.append(f"#{i}: visible option without box/center")
                    else:
                        if not (b["x"] <= c["x"] <= b["x"] + b["w"] and b["y"] <= c["y"] <= b["y"] + b["h"]):
                            errs.append(f"#{i}: option center outside box")
    if errs:
        print("Validation failed:")
        for e in errs[:20]:
            print(" -", e)
        return False
    print("Validation OK")
    return True
