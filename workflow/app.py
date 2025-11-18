import dearpygui.dearpygui as dpg
from PIL import Image, ImageDraw
import numpy as np
import os
import json as pyjson
import math
import time
import threading

WINDOW_TAG = "main_window"
CANVAS_TAG = "canvas"
TEXREG_TAG = "texture_registry"
JSON_VIEWER_TAG = "json_viewer"
JSON_TEXT_TAG = "json_viewer_text"
NOTE_EDITOR_TAG = "note_editor"
NOTE_TEXT_TAG = "note_editor_text"
STATE_FILE = "flowui_state.json"

# Stan
screens = []
json_objects = []
notes = []
screen_counter = 0
texture_counter = 0

lines = []

pan_x = 0.0
pan_y = 0.0
zoom = 1.0

active_object_id = None
drag_offset = (0.0, 0.0)

is_panning = False
last_mouse_pos = (0.0, 0.0)

# Resize
resize_mode = False
resize_object_id = None
resize_edge_right = False
resize_edge_bottom = False
MIN_SIZE = 50

# Rysowanie linii (D)
draw_line_mode = False
line_start = None

# Gumka do linii (C)
erase_mode = False
ERASE_DISTANCE_PX = 15

# Auto-reload
auto_reload_enabled = True
AUTO_RELOAD_INTERVAL = 2

# Tryb dodawania notatki (N)
add_note_mode = False

# Aktywna notatka do edycji
active_note_id = None

# Flaga - czy trzeba zaktualizować teksturę po resize
needs_texture_update = False


# ================== POMOCNICZE ==================

def world_from_screen(mx, my):
    """Screen -> world coords."""
    cx, cy = dpg.get_item_rect_min(CANVAS_TAG)
    local_mx = mx - cx
    local_my = my - cy
    wx = (local_mx - pan_x) / zoom
    wy = (local_my - pan_y) / zoom
    return wx, wy


def distance_point_to_line_segment(px, py, x1, y1, x2, y2):
    """Odległość punktu od odcinka."""
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        return math.sqrt((px - x1)**2 + (py - y1)**2)
    
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    
    return math.sqrt((px - proj_x)**2 + (py - proj_y)**2)


def create_json_texture(filename, width=200, height=150):
    """Tworzy teksturę dla JSON."""
    img = Image.new("RGBA", (width, height), (255, 220, 100, 255))
    draw = ImageDraw.Draw(img)
    
    draw.rectangle([0, 0, width-1, height-1], outline=(200, 150, 0, 255), width=3)
    draw.rectangle([10, 10, width-10, 50], fill=(255, 200, 50, 255))
    
    try:
        draw.text((width//2 - 30, 18), "{ JSON }", fill=(0, 0, 0, 255))
    except:
        pass
    
    short_name = filename[:25] if len(filename) <= 25 else filename[:22] + "..."
    try:
        draw.text((15, 60), short_name, fill=(50, 50, 50, 255))
    except:
        pass
    
    for i in range(4):
        y = 85 + i * 12
        line_width = width - 30 - (i % 2) * 40
        draw.rectangle([15, y, line_width, y+6], fill=(255, 255, 200, 255))
    
    draw.ellipse([width-35, height-35, width-10, height-10], fill=(100, 200, 100, 255))
    
    return img


def create_note_texture(text, width=250, height=200):
    """Tworzy teksturę dla notatki/bloczku."""
    width = max(100, int(width))
    height = max(80, int(height))
    
    img = Image.new("RGBA", (width, height), (173, 216, 230, 255))
    draw = ImageDraw.Draw(img)
    
    draw.rectangle([0, 0, width-1, height-1], outline=(100, 150, 180, 255), width=3)
    
    header_height = min(35, height // 4)
    draw.rectangle([5, 5, width-5, header_height], fill=(135, 206, 250, 255))
    
    try:
        draw.text((10, 10), "📝 Notatka", fill=(0, 0, 0, 255))
    except:
        draw.text((10, 10), "NOTE", fill=(0, 0, 0, 255))
    
    lines_text = []
    current_line = ""
    words = text.split()
    
    max_chars = max(15, width // 9)
    
    for word in words:
        if len(current_line) + len(word) + 1 <= max_chars:
            current_line += word + " "
        else:
            if current_line:
                lines_text.append(current_line.strip())
            current_line = word + " "
    
    if current_line:
        lines_text.append(current_line.strip())
    
    y_offset = header_height + 10
    line_height = 14
    max_lines = (height - header_height - 30) // line_height
    
    for i, line in enumerate(lines_text[:max_lines]):
        try:
            draw.text((10, y_offset + i * line_height), line, fill=(0, 0, 0, 255))
        except:
            pass
    
    if len(lines_text) > max_lines:
        try:
            draw.text((10, height - 20), "...", fill=(100, 100, 100, 255))
        except:
            pass
    
    if width > 40 and height > 40:
        draw.ellipse([width-30, height-30, width-5, height-5], fill=(100, 200, 255, 255))
        try:
            draw.text((width-25, height-25), "✎", fill=(255, 255, 255, 255))
        except:
            pass
    
    return img


# ================== RYSOWANIE ==================

def redraw_canvas():
    """Przerysuj canvas."""
    if not dpg.does_item_exist(CANVAS_TAG):
        return

    dpg.delete_item(CANVAS_TAG, children_only=True)

    step = 50
    size = 5000
    for x in range(-size, size + 1, step):
        x1 = pan_x + x * zoom
        y1 = pan_y - size * zoom
        x2 = pan_x + x * zoom
        y2 = pan_y + size * zoom
        dpg.draw_line((x1, y1), (x2, y2), color=(40, 40, 40, 255), parent=CANVAS_TAG)
    
    for y in range(-size, size + 1, step):
        x1 = pan_x - size * zoom
        y1 = pan_y + y * zoom
        x2 = pan_x + size * zoom
        y2 = pan_y + y * zoom
        dpg.draw_line((x1, y1), (x2, y2), color=(40, 40, 40, 255), parent=CANVAS_TAG)

    dpg.draw_line((pan_x - size * zoom, pan_y), (pan_x + size * zoom, pan_y),
                  color=(120, 120, 255, 255), thickness=2, parent=CANVAS_TAG)
    dpg.draw_line((pan_x, pan_y - size * zoom), (pan_x, pan_y + size * zoom),
                  color=(120, 120, 255, 255), thickness=2, parent=CANVAS_TAG)

    for s in screens:
        x1 = pan_x + s["x"] * zoom
        y1 = pan_y + s["y"] * zoom
        x2 = x1 + s["w"] * zoom
        y2 = y1 + s["h"] * zoom
        dpg.draw_image(s["tex"], (x1, y1), (x2, y2), parent=CANVAS_TAG)

    for j in json_objects:
        x1 = pan_x + j["x"] * zoom
        y1 = pan_y + j["y"] * zoom
        x2 = x1 + j["w"] * zoom
        y2 = y1 + j["h"] * zoom
        dpg.draw_image(j["tex"], (x1, y1), (x2, y2), parent=CANVAS_TAG)

    for n in notes:
        x1 = pan_x + n["x"] * zoom
        y1 = pan_y + n["y"] * zoom
        x2 = x1 + n["w"] * zoom
        y2 = y1 + n["h"] * zoom
        dpg.draw_image(n["tex"], (x1, y1), (x2, y2), parent=CANVAS_TAG)

    for ln in lines:
        x1 = pan_x + ln["x1"] * zoom
        y1 = pan_y + ln["y1"] * zoom
        x2 = pan_x + ln["x2"] * zoom
        y2 = pan_y + ln["y2"] * zoom
        
        color = (0, 255, 0, 255)
        thickness = 2
        
        if erase_mode:
            mx, my = dpg.get_mouse_pos()
            wx, wy = world_from_screen(mx, my)
            dist_world = distance_point_to_line_segment(wx, wy, ln["x1"], ln["y1"], ln["x2"], ln["y2"])
            dist_screen = dist_world * zoom
            
            if dist_screen <= ERASE_DISTANCE_PX:
                color = (255, 50, 50, 255)
                thickness = 4
        
        dpg.draw_line((x1, y1), (x2, y2), color=color, thickness=thickness, parent=CANVAS_TAG)

    if draw_line_mode and line_start is not None:
        mx, my = dpg.get_mouse_pos()
        wx, wy = world_from_screen(mx, my)
        x1 = pan_x + line_start[0] * zoom
        y1 = pan_y + line_start[1] * zoom
        x2 = pan_x + wx * zoom
        y2 = pan_y + wy * zoom
        dpg.draw_line((x1, y1), (x2, y2), color=(0, 200, 0, 150), thickness=1, parent=CANVAS_TAG)


def add_screen_from_path(path, x=0.0, y=0.0):
    """Dodaje PNG jako screen."""
    global screen_counter, texture_counter

    if not os.path.exists(path):
        msg = f"❌ Plik nie istnieje: {path}"
        print(msg)
        dpg.set_value("result_text", msg)
        return

    image = Image.open(path).convert("RGBA")
    w, h = image.size

    raw = np.frombuffer(image.tobytes(), dtype=np.uint8).astype(np.float32) / 255.0
    data = raw.tolist()

    tex_tag = f"tex_{texture_counter}"
    texture_counter += 1

    dpg.add_static_texture(w, h, data, tag=tex_tag, parent=TEXREG_TAG)

    screen_id = f"screen_{screen_counter}"
    screen_counter += 1

    screens.append({
        "id": screen_id,
        "tex": tex_tag,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "path": path,
        "type": "png",
        "mtime": os.path.getmtime(path)
    })

    info = f"✅ PNG: {os.path.basename(path)} ({w}x{h})"
    print(info)
    dpg.set_value("result_text", info)
    redraw_canvas()


def add_json_to_canvas(path, x=0.0, y=0.0):
    """Dodaje JSON jako graficzny element."""
    global screen_counter, texture_counter

    if not os.path.exists(path):
        msg = f"❌ JSON nie istnieje: {path}"
        print(msg)
        dpg.set_value("result_text", msg)
        return

    filename = os.path.basename(path)
    img = create_json_texture(filename, width=200, height=150)
    w, h = img.size

    raw = np.frombuffer(img.tobytes(), dtype=np.uint8).astype(np.float32) / 255.0
    data = raw.tolist()

    tex_tag = f"tex_{texture_counter}"
    texture_counter += 1

    dpg.add_static_texture(w, h, data, tag=tex_tag, parent=TEXREG_TAG)

    json_id = f"json_{screen_counter}"
    screen_counter += 1

    json_objects.append({
        "id": json_id,
        "tex": tex_tag,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "path": path,
        "type": "json",
        "mtime": os.path.getmtime(path)
    })

    info = f"✅ JSON: {filename}"
    print(info)
    dpg.set_value("result_text", info)
    redraw_canvas()


def add_note_to_canvas(x=0.0, y=0.0, text="Nowa notatka"):
    """Dodaje notatkę na canvas."""
    global screen_counter, texture_counter

    img = create_note_texture(text, width=250, height=200)
    w, h = img.size

    raw = np.frombuffer(img.tobytes(), dtype=np.uint8).astype(np.float32) / 255.0
    data = raw.tolist()

    tex_tag = f"tex_{texture_counter}"
    texture_counter += 1

    dpg.add_static_texture(w, h, data, tag=tex_tag, parent=TEXREG_TAG)

    note_id = f"note_{screen_counter}"
    screen_counter += 1

    notes.append({
        "id": note_id,
        "tex": tex_tag,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "text": text,
        "type": "note"
    })

    info = f"✅ Notatka: {text[:20]}..."
    print(info)
    dpg.set_value("result_text", info)
    redraw_canvas()


def update_note_texture(note):
    """Aktualizuje teksturę notatki po edycji."""
    global texture_counter
    
    img = create_note_texture(note["text"], width=int(note["w"]), height=int(note["h"]))
    w, h = img.size
    
    raw = np.frombuffer(img.tobytes(), dtype=np.uint8).astype(np.float32) / 255.0
    data = raw.tolist()
    
    old_tex = note["tex"]
    if dpg.does_item_exist(old_tex):
        dpg.delete_item(old_tex)
    
    new_tex_tag = f"tex_note_{texture_counter}"
    texture_counter += 1
    
    dpg.add_static_texture(w, h, data, tag=new_tex_tag, parent=TEXREG_TAG)
    
    note["tex"] = new_tex_tag
    redraw_canvas()


def delete_note(note_id):
    """Usuń notatkę."""
    global notes
    
    for i, note in enumerate(notes):
        if note["id"] == note_id:
            if dpg.does_item_exist(note["tex"]):
                dpg.delete_item(note["tex"])
            
            notes.pop(i)
            print(f"🗑️ Usunięto notatkę: {note['text'][:30]}...")
            
            if dpg.does_item_exist(NOTE_EDITOR_TAG):
                dpg.hide_item(NOTE_EDITOR_TAG)
            
            redraw_canvas()
            dpg.set_value("result_text", "🗑️ Notatka usunięta")
            return
    
    print(f"❌ Nie znaleziono notatki: {note_id}")


def open_note_editor(note_id):
    """Otwórz edytor notatki."""
    global active_note_id
    
    note = None
    for n in notes:
        if n["id"] == note_id:
            note = n
            break
    
    if not note:
        return
    
    active_note_id = note_id
    
    if not dpg.does_item_exist(NOTE_EDITOR_TAG):
        with dpg.window(label=f"Edycja notatki", tag=NOTE_EDITOR_TAG,
                        width=500, height=450, pos=(100, 100), modal=True, no_collapse=True):
            dpg.add_text("Edytuj tekst notatki:")
            dpg.add_input_text(tag=NOTE_TEXT_TAG,
                               default_value=note["text"],
                               multiline=True,
                               width=-1, height=300)
            with dpg.group(horizontal=True):
                dpg.add_button(label="💾 Zapisz", callback=save_note_edit, width=120)
                dpg.add_button(label="🗑️ Usuń notatkę", callback=lambda: delete_note(active_note_id), width=120)
                dpg.add_button(label="❌ Anuluj", callback=lambda: dpg.hide_item(NOTE_EDITOR_TAG), width=120)
    else:
        dpg.configure_item(NOTE_EDITOR_TAG, label=f"Edycja notatki", show=True)
        dpg.set_value(NOTE_TEXT_TAG, note["text"])
    
    print(f"📝 Edycja notatki: {note_id}")


def save_note_edit():
    """Zapisz edycję notatki."""
    global active_note_id
    
    if not active_note_id:
        return
    
    new_text = dpg.get_value(NOTE_TEXT_TAG)
    
    for n in notes:
        if n["id"] == active_note_id:
            n["text"] = new_text
            update_note_texture(n)
            print(f"💾 Zaktualizowano notatkę: {new_text[:30]}...")
            break
    
    dpg.hide_item(NOTE_EDITOR_TAG)
    active_note_id = None


# ================== HOT RELOAD ==================

def check_and_reload_files():
    """Sprawdza czy pliki się zmieniły i przeładowuje."""
    global texture_counter
    
    if not dpg.does_item_exist(CANVAS_TAG):
        return
    
    reloaded = []
    needs_redraw = False
    
    for s in screens:
        if not os.path.exists(s["path"]):
            continue
        
        try:
            current_mtime = os.path.getmtime(s["path"])
            if current_mtime > s["mtime"]:
                print(f"🔄 Przeładowuję PNG: {os.path.basename(s['path'])}")
                
                image = Image.open(s["path"]).convert("RGBA")
                w, h = image.size
                raw = np.frombuffer(image.tobytes(), dtype=np.uint8).astype(np.float32) / 255.0
                data = raw.tolist()
                
                old_tex = s["tex"]
                if dpg.does_item_exist(old_tex):
                    dpg.delete_item(old_tex)
                
                new_tex_tag = f"tex_reload_{texture_counter}"
                texture_counter += 1
                
                dpg.add_static_texture(w, h, data, tag=new_tex_tag, parent=TEXREG_TAG)
                
                s["tex"] = new_tex_tag
                s["w"] = w
                s["h"] = h
                s["mtime"] = current_mtime
                
                reloaded.append(os.path.basename(s["path"]))
                needs_redraw = True
                
        except Exception as e:
            print(f"❌ Błąd przeładowania PNG {s['path']}: {e}")
    
    for j in json_objects:
        if not os.path.exists(j["path"]):
            continue
        
        try:
            current_mtime = os.path.getmtime(j["path"])
            if current_mtime > j["mtime"]:
                print(f"🔄 Przeładowuję JSON: {os.path.basename(j['path'])}")
                
                filename = os.path.basename(j["path"])
                img = create_json_texture(filename, width=int(j["w"]), height=int(j["h"]))
                w, h = img.size
                raw = np.frombuffer(img.tobytes(), dtype=np.uint8).astype(np.float32) / 255.0
                data = raw.tolist()
                
                old_tex = j["tex"]
                if dpg.does_item_exist(old_tex):
                    dpg.delete_item(old_tex)
                
                new_tex_tag = f"tex_reload_{texture_counter}"
                texture_counter += 1
                
                dpg.add_static_texture(w, h, data, tag=new_tex_tag, parent=TEXREG_TAG)
                
                j["tex"] = new_tex_tag
                j["mtime"] = current_mtime
                
                reloaded.append(os.path.basename(j["path"]))
                needs_redraw = True
                
        except Exception as e:
            print(f"❌ Błąd przeładowania JSON {j['path']}: {e}")
    
    if needs_redraw:
        redraw_canvas()
        msg = f"🔄 Odświeżono: {', '.join(reloaded)}"
        dpg.set_value("result_text", msg)


def auto_reload_thread():
    """Wątek sprawdzający zmiany."""
    while True:
        time.sleep(AUTO_RELOAD_INTERVAL)
        if auto_reload_enabled:
            try:
                check_and_reload_files()
            except Exception as e:
                print(f"❌ Błąd auto-reload: {e}")


def toggle_auto_reload():
    """Przełącz auto-reload."""
    global auto_reload_enabled
    auto_reload_enabled = not auto_reload_enabled
    status = "ON" if auto_reload_enabled else "OFF"
    color = (100, 255, 100) if auto_reload_enabled else (255, 100, 100)
    dpg.configure_item("auto_reload_status", default_value=f"Auto: {status}", color=color)
    print(f"🔄 Auto-reload: {status}")


def open_json_viewer(path):
    """Otwórz viewer JSON."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        try:
            obj = pyjson.loads(content)
            pretty = pyjson.dumps(obj, ensure_ascii=False, indent=2)
        except:
            pretty = content
    except Exception as e:
        pretty = f"❌ Błąd: {e}"

    title = f"JSON: {os.path.basename(path)}"

    if not dpg.does_item_exist(JSON_VIEWER_TAG):
        with dpg.window(label=title, tag=JSON_VIEWER_TAG, width=600, height=700, pos=(50, 50)):
            dpg.add_input_text(tag=JSON_TEXT_TAG, default_value=pretty,
                             multiline=True, readonly=True, width=-1, height=-1)
    else:
        dpg.configure_item(JSON_VIEWER_TAG, label=title, show=True)
        dpg.set_value(JSON_TEXT_TAG, pretty)
    
    print(f"📖 Otwarto JSON: {os.path.basename(path)}")


# ================== ZAPIS/ODCZYT ==================

def save_state():
    """Zapisz stan."""
    try:
        state = {
            "pan_x": pan_x,
            "pan_y": pan_y,
            "zoom": zoom,
            "screens": [{"path": s["path"], "x": s["x"], "y": s["y"], "w": s["w"], "h": s["h"]} for s in screens],
            "json_objects": [{"path": j["path"], "x": j["x"], "y": j["y"], "w": j["w"], "h": j["h"]} for j in json_objects],
            "notes": [{"text": n["text"], "x": n["x"], "y": n["y"], "w": n["w"], "h": n["h"]} for n in notes],
            "lines": lines
        }
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            pyjson.dump(state, f, ensure_ascii=False, indent=2)
        print(f"💾 Stan zapisany")
    except Exception as e:
        print(f"❌ Błąd zapisu: {e}")


def load_state():
    """Wczytaj stan."""
    global pan_x, pan_y, zoom, lines

    if not os.path.exists(STATE_FILE):
        return False

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = pyjson.load(f)
    except Exception as e:
        print(f"❌ Błąd odczytu: {e}")
        return False

    pan_x = float(state.get("pan_x", 0.0))
    pan_y = float(state.get("pan_y", 0.0))
    zoom = max(0.1, min(5.0, float(state.get("zoom", 1.0))))

    for s in state.get("screens", []):
        path = s.get("path")
        if path and os.path.exists(path):
            add_screen_from_path(path, x=float(s.get("x", 0.0)), y=float(s.get("y", 0.0)))
            screens[-1]["w"] = float(s.get("w", screens[-1]["w"]))
            screens[-1]["h"] = float(s.get("h", screens[-1]["h"]))

    for j in state.get("json_objects", []):
        path = j.get("path")
        if path and os.path.exists(path):
            add_json_to_canvas(path, x=float(j.get("x", 0.0)), y=float(j.get("y", 0.0)))
            json_objects[-1]["w"] = float(j.get("w", json_objects[-1]["w"]))
            json_objects[-1]["h"] = float(j.get("h", json_objects[-1]["h"]))

    for n in state.get("notes", []):
        text = n.get("text", "")
        add_note_to_canvas(x=float(n.get("x", 0.0)), y=float(n.get("y", 0.0)), text=text)
        notes[-1]["w"] = float(n.get("w", notes[-1]["w"]))
        notes[-1]["h"] = float(n.get("h", notes[-1]["h"]))

    for ln in state.get("lines", []):
        if all(k in ln for k in ("x1", "y1", "x2", "y2")):
            lines.append({"x1": float(ln["x1"]), "y1": float(ln["y1"]), 
                         "x2": float(ln["x2"]), "y2": float(ln["y2"])})

    redraw_canvas()
    print("📂 Stan wczytany")
    return True


def on_exit():
    save_state()


# ================== CALLBACKI ==================

def get_all_objects():
    return screens + json_objects + notes


def on_file_dialog(sender, app_data):
    if not app_data:
        return
    path = app_data.get("file_path_name")
    if not path:
        return

    lower = path.lower()
    if lower.endswith(".png"):
        add_screen_from_path(path)
    elif lower.endswith(".json"):
        add_json_to_canvas(path)
    else:
        dpg.set_value("result_text", f"Nieobsługiwany typ: {path}")


def on_file_button():
    dpg.show_item("file_dialog_id")


def on_key_d(sender, app_data):
    global draw_line_mode, line_start, erase_mode, add_note_mode
    
    if erase_mode:
        erase_mode = False
        dpg.set_value("erase_text", "Gumka (C): OFF")
    
    if add_note_mode:
        add_note_mode = False
        dpg.set_value("note_text", "Notatka (N): OFF")
    
    draw_line_mode = not draw_line_mode
    line_start = None
    mode = "ON" if draw_line_mode else "OFF"
    dpg.set_value("mode_text", f"Tryb linii (D): {mode}")
    print(f"🖊️ Rysowanie: {mode}")
    redraw_canvas()


def on_key_c(sender, app_data):
    global erase_mode, draw_line_mode, line_start, add_note_mode
    
    if draw_line_mode:
        draw_line_mode = False
        line_start = None
        dpg.set_value("mode_text", "Tryb linii (D): OFF")
    
    if add_note_mode:
        add_note_mode = False
        dpg.set_value("note_text", "Notatka (N): OFF")
    
    erase_mode = not erase_mode
    mode = "ON" if erase_mode else "OFF"
    dpg.set_value("erase_text", f"Gumka (C): {mode}")
    print(f"🧹 Gumka: {mode}")
    redraw_canvas()


def on_key_n(sender, app_data):
    """Tryb dodawania notatki."""
    global add_note_mode, draw_line_mode, erase_mode, line_start
    
    if draw_line_mode:
        draw_line_mode = False
        line_start = None
        dpg.set_value("mode_text", "Tryb linii (D): OFF")
    
    if erase_mode:
        erase_mode = False
        dpg.set_value("erase_text", "Gumka (C): OFF")
    
    add_note_mode = not add_note_mode
    mode = "ON" if add_note_mode else "OFF"
    dpg.set_value("note_text", f"Notatka (N): {mode}")
    print(f"📝 Dodawanie notatki: {mode}")


def on_left_click(sender, app_data):
    global active_object_id, drag_offset
    global resize_mode, resize_object_id, resize_edge_right, resize_edge_bottom
    global line_start, add_note_mode, needs_texture_update

    mx, my = dpg.get_mouse_pos()
    wx, wy = world_from_screen(mx, my)

    # PRIORYTET 0: Dodawanie notatki
    if add_note_mode:
        add_note_to_canvas(x=wx, y=wy, text="Nowa notatka\nKliknij dwukrotnie aby edytować")
        add_note_mode = False
        dpg.set_value("note_text", "Notatka (N): OFF")
        return

    # PRIORYTET 1: Gumka
    if erase_mode:
        removed_count = 0
        i = 0
        while i < len(lines):
            ln = lines[i]
            dist_world = distance_point_to_line_segment(wx, wy, ln["x1"], ln["y1"], ln["x2"], ln["y2"])
            dist_screen = dist_world * zoom
            
            if dist_screen <= ERASE_DISTANCE_PX:
                removed = lines.pop(i)
                print(f"🗑️ Usunięto linię")
                removed_count += 1
            else:
                i += 1
        
        if removed_count > 0:
            redraw_canvas()
        return

    # PRIORYTET 2: Rysowanie linii
    if draw_line_mode:
        if line_start is None:
            line_start = (wx, wy)
        else:
            lines.append({"x1": line_start[0], "y1": line_start[1], "x2": wx, "y2": wy})
            line_start = None
        redraw_canvas()
        return

    # PRIORYTET 3: Resize/Drag - POPRAWIONA WERSJA
    active_object_id = None
    resize_mode = False
    resize_object_id = None
    resize_edge_right = False
    resize_edge_bottom = False
    needs_texture_update = False

    # Większy margines dla lepszego wykrywania
    RESIZE_MARGIN_WORLD = 20 / max(zoom, 0.001)  # 20 pikseli w world units

    all_objs = get_all_objects()
    for obj in reversed(all_objs):
        sx, sy, sw, sh = obj["x"], obj["y"], obj["w"], obj["h"]
        
        # Sprawdź czy jesteśmy blisko prawej krawędzi
        near_right = abs(wx - (sx + sw)) <= RESIZE_MARGIN_WORLD
        # Sprawdź czy jesteśmy blisko dolnej krawędzi
        near_bottom = abs(wy - (sy + sh)) <= RESIZE_MARGIN_WORLD
        
        # Sprawdź czy jesteśmy w obszarze obiektu (z marginesem resize)
        in_x_range = (sx - RESIZE_MARGIN_WORLD) <= wx <= (sx + sw + RESIZE_MARGIN_WORLD)
        in_y_range = (sy - RESIZE_MARGIN_WORLD) <= wy <= (sy + sh + RESIZE_MARGIN_WORLD)
        
        # Jeśli jesteśmy blisko krawędzi i w zakresie obiektu
        if in_x_range and in_y_range and (near_right or near_bottom):
            resize_mode = True
            resize_object_id = obj["id"]
            resize_edge_right = near_right
            resize_edge_bottom = near_bottom
            
            # Debug
            print(f"🔧 RESIZE MODE: {obj['id']}")
            print(f"   Pozycja myszy: wx={wx:.1f}, wy={wy:.1f}")
            print(f"   Obiekt: x={sx:.1f}, y={sy:.1f}, w={sw:.1f}, h={sh:.1f}")
            print(f"   Prawa krawędź: {sx + sw:.1f}, odległość: {abs(wx - (sx + sw)):.1f}")
            print(f"   Dolna krawędź: {sy + sh:.1f}, odległość: {abs(wy - (sy + sh)):.1f}")
            print(f"   Resize RIGHT: {resize_edge_right}, BOTTOM: {resize_edge_bottom}")
            print(f"   Margin: {RESIZE_MARGIN_WORLD:.1f}")
            
            dpg.set_value("result_text", f"🔧 Resize: {'prawo' if near_right else ''} {'dół' if near_bottom else ''}")
            return

        # Sprawdź czy kliknięto WEWNĄTRZ obiektu (dla drag)
        inside = sx <= wx <= sx + sw and sy <= wy <= sy + sh
        
        if inside:
            active_object_id = obj["id"]
            drag_offset = (wx - sx, wy - sy)
            print(f"✋ DRAG MODE: {obj['id']}")
            dpg.set_value("result_text", f"✋ Przeciąganie: {obj['id']}")
            return


def on_left_double_click(sender, app_data):
    mx, my = dpg.get_mouse_pos()
    wx, wy = world_from_screen(mx, my)

    for obj in reversed(notes):
        sx, sy, sw, sh = obj["x"], obj["y"], obj["w"], obj["h"]
        inside = sx <= wx <= sx + sw and sy <= wy <= sy + sh

        if inside:
            open_note_editor(obj["id"])
            return

    for obj in reversed(json_objects):
        sx, sy, sw, sh = obj["x"], obj["y"], obj["w"], obj["h"]
        inside = sx <= wx <= sx + sw and sy <= wy <= sy + sh

        if inside:
            open_json_viewer(obj["path"])
            return


def on_left_release(sender, app_data):
    global active_object_id, resize_mode, resize_object_id, needs_texture_update
    
    if resize_mode and resize_object_id and needs_texture_update:
        for obj in notes:
            if obj["id"] == resize_object_id:
                update_note_texture(obj)
                break
        needs_texture_update = False
    
    active_object_id = None
    resize_mode = False
    resize_object_id = None


def on_left_drag(sender, app_data):
    global needs_texture_update
    
    mx, my = dpg.get_mouse_pos()
    wx, wy = world_from_screen(mx, my)

    # RESIZE MODE
    if resize_mode and resize_object_id is not None:
        all_objs = get_all_objects()
        for obj in all_objs:
            if obj["id"] == resize_object_id:
                old_w = obj["w"]
                old_h = obj["h"]
                
                # Aktualizuj szerokość
                if resize_edge_right:
                    new_w = max(MIN_SIZE, wx - obj["x"])
                    obj["w"] = new_w
                
                # Aktualizuj wysokość
                if resize_edge_bottom:
                    new_h = max(MIN_SIZE, wy - obj["y"])
                    obj["h"] = new_h
                
                # Debug - pokaż co się zmienia
                if obj["w"] != old_w or obj["h"] != old_h:
                    print(f"📏 Resize: {old_w:.0f}x{old_h:.0f} -> {obj['w']:.0f}x{obj['h']:.0f}")
                
                # Oznacz że trzeba zaktualizować teksturę notatki
                if obj["type"] == "note":
                    needs_texture_update = True
                
                break
        
        redraw_canvas()
        return

    # DRAG MODE
    if active_object_id is None:
        return

    all_objs = get_all_objects()
    for obj in all_objs:
        if obj["id"] == active_object_id:
            obj["x"] = wx - drag_offset[0]
            obj["y"] = wy - drag_offset[1]
            break

    redraw_canvas()


def on_middle_down(sender, app_data):
    global is_panning, last_mouse_pos
    is_panning = True
    last_mouse_pos = dpg.get_mouse_pos()


def on_middle_drag(sender, app_data):
    global pan_x, pan_y, last_mouse_pos
    if not is_panning:
        return
    
    mx, my = dpg.get_mouse_pos()
    dx = mx - last_mouse_pos[0]
    dy = my - last_mouse_pos[1]
    pan_x += dx
    pan_y += dy
    last_mouse_pos = (mx, my)
    redraw_canvas()


def on_middle_release(sender, app_data):
    global is_panning
    is_panning = False


def on_mouse_wheel(sender, app_data):
    global zoom, pan_x, pan_y

    delta = app_data
    if delta == 0:
        return

    old_zoom = zoom
    factor = 1.1 if delta > 0 else 1 / 1.1
    zoom = max(0.1, min(5.0, zoom * factor))

    mx, my = dpg.get_mouse_pos()
    if zoom != old_zoom:
        pan_x = mx - (mx - pan_x) * (zoom / old_zoom)
        pan_y = my - (my - pan_y) * (zoom / old_zoom)

    redraw_canvas()


def on_mouse_move(sender, app_data):
    if (draw_line_mode and line_start is not None) or erase_mode:
        redraw_canvas()


# ================== UI ==================

dpg.create_context()

with dpg.texture_registry(tag=TEXREG_TAG):
    pass

with dpg.window(label="Flow GUI", tag=WINDOW_TAG, width=1200, height=800):
    dpg.add_text("🗺️ Canvas: PNG + JSON + Notatki + Linie + Pan/Zoom + Gumka")

    with dpg.group(horizontal=True):
        dpg.add_button(label="📁 Dodaj PNG/JSON", callback=on_file_button)
        dpg.add_button(label="🔄 Odśwież", callback=lambda: check_and_reload_files())
        dpg.add_button(label="⏯️", callback=toggle_auto_reload, width=30)
        dpg.add_text("Auto: ON", tag="auto_reload_status", color=(100, 255, 100))
        dpg.add_text("Gotowy", tag="result_text")

    with dpg.group(horizontal=True):
        dpg.add_text("Tryb linii (D): OFF", tag="mode_text")
        dpg.add_text("  |  ", color=(100, 100, 100))
        dpg.add_text("Gumka (C): OFF", tag="erase_text")
        dpg.add_text("  |  ", color=(100, 100, 100))
        dpg.add_text("Notatka (N): OFF", tag="note_text")

    dpg.add_separator()
    dpg.add_text("LPM: drag/resize | DWUKLIK: edytuj notatkę/JSON | PPM: pan | Scroll: zoom")
    dpg.add_text("D: rysuj linie | C: gumka | N: dodaj notatkę (kliknij gdzie ma być)")
    dpg.add_separator()

    dpg.add_drawlist(width=1100, height=550, tag=CANVAS_TAG)

with dpg.file_dialog(
    directory_selector=False,
    show=False,
    callback=on_file_dialog,
    id="file_dialog_id",
    width=700,
    height=400
):
    dpg.add_file_extension(".*", color=(200, 200, 200, 255))
    dpg.add_file_extension(".png", color=(150, 255, 150, 255))
    dpg.add_file_extension(".json", color=(255, 200, 0, 255))

with dpg.handler_registry():
    dpg.add_mouse_click_handler(button=0, callback=on_left_click)
    dpg.add_mouse_double_click_handler(button=0, callback=on_left_double_click)
    dpg.add_mouse_release_handler(button=0, callback=on_left_release)
    dpg.add_mouse_drag_handler(button=0, threshold=2, callback=on_left_drag)

    dpg.add_mouse_click_handler(button=1, callback=on_middle_down)
    dpg.add_mouse_release_handler(button=1, callback=on_middle_release)
    dpg.add_mouse_drag_handler(button=1, threshold=0, callback=on_middle_drag)

    dpg.add_mouse_wheel_handler(callback=on_mouse_wheel)
    dpg.add_mouse_move_handler(callback=on_mouse_move)

    dpg.add_key_press_handler(key=dpg.mvKey_D, callback=on_key_d)
    dpg.add_key_press_handler(key=dpg.mvKey_C, callback=on_key_c)
    dpg.add_key_press_handler(key=dpg.mvKey_N, callback=on_key_n)

dpg.create_viewport(title="Flow UI - Canvas", width=1200, height=800)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_exit_callback(on_exit)

if not load_state():
    default_image_path = r"E:\BOT ANK\bot\moje_AI\yolov8\FULL BOT\dom_live\debug\ocr_strip1_active.png"
    if os.path.exists(default_image_path):
        add_screen_from_path(default_image_path, x=0.0, y=0.0)

reload_thread = threading.Thread(target=auto_reload_thread, daemon=True)
reload_thread.start()

try:
    dpg.start_dearpygui()
except KeyboardInterrupt:
    print("\n⚠️ Ctrl+C")
    save_state()
finally:
    dpg.destroy_context()