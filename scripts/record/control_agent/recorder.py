import json
import math
import threading
import time
import platform
import os
from collections import deque
from pynput import keyboard

# Konfiguracja
sample_interval = 0.0015       # s; ciągłe próbkowanie kursora (np. 0.001–0.002 to ~500–1000 Hz)
slider_threshold = 0.05        # s; >= oznacza slider
output_file = "trajectory.json"
exit_key = keyboard.Key.f12

# Bufor próbek (ile sek. historii trzymać do wycinania segmentów Z/X)
max_buffer_seconds = 20.0
# batching próbek do jednej operacji na lock (mniejsza kolizja wątków)
batch_flush_time = 0.004       # s
batch_flush_size = 6

# --- Pobieranie pozycji kursora (Windows: GetCursorPos; fallback dla innych OS) ---
if platform.system() == "Windows":
    import ctypes
    from ctypes import wintypes

    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    _user32 = ctypes.windll.user32
    try:
        _timeBeginPeriod = ctypes.windll.winmm.timeBeginPeriod
        _timeEndPeriod = ctypes.windll.winmm.timeEndPeriod
    except Exception:
        _timeBeginPeriod = None
        _timeEndPeriod = None

    def get_cursor_pos():
        pt = POINT()
        _user32.GetCursorPos(ctypes.byref(pt))
        return int(pt.x), int(pt.y)

    def enable_high_res_timer():
        if _timeBeginPeriod:
            try:
                _timeBeginPeriod(1)
            except Exception:
                pass

    def disable_high_res_timer():
        if _timeEndPeriod:
            try:
                _timeEndPeriod(1)
            except Exception:
                pass
else:
    # Fallback (bezpośredni odczyt z pynput Controller)
    from pynput import mouse
    _mc = mouse.Controller()

    def get_cursor_pos():
        x, y = _mc.position
        return int(x), int(y)

    def enable_high_res_timer():
        pass

    def disable_high_res_timer():
        pass

# --- Blokady i zdarzenia ---
samples_lock = threading.Lock()
data_lock = threading.Lock()
stop_event = threading.Event()
save_event = threading.Event()

# --- Dane ciągłe i segmenty ---
max_samples = max(1, int(max_buffer_seconds / sample_interval) + 1000)
samples = deque(maxlen=max_samples)   # elementy: (t_abs, x, y)

# ZMIANA: Przechowujemy tylko NOWE trajektorie w sesji
new_trajectory_data = []
existing_count = 0  # Liczba trajektorii już w pliku
session_saved_count = 0  # Ile z nowych już zapisaliśmy

active_key = None               # 'z' albo 'x' jeśli segment otwarty
segment_t0 = None               # perf_counter() rozpoczęcia segmentu
segment_pos0 = None             # (x,y) przy starcie segmentu

# --- Narzędzia ---
def normalize_traj(traj_abs, t0):
    # traj_abs: [(t_abs, x, y), ...] -> [[x, y, t_rel], ...]
    return [[x, y, round(float(t - t0), 6)] for (t, x, y) in traj_abs]

def calculate_traj_info(traj_rel, key_used):
    if not traj_rel:
        return {
            "length": 0.0, "angle": 0.0, "direction": "none",
            "duration": 0.0, "points": 0, "key": key_used, "is_slider": False
        }
    dx = traj_rel[-1][0] - traj_rel[0][0]
    dy = traj_rel[-1][1] - traj_rel[0][1]
    length = math.hypot(dx, dy)
    angle = math.degrees(math.atan2(dy, dx)) if length > 0 else 0.0
    direction = "right" if dx > 0 else "left" if dx < 0 else "none"
    duration = traj_rel[-1][2] - traj_rel[0][2]
    return {
        "length": float(length),
        "angle": float(angle),
        "direction": direction,
        "duration": float(duration),
        "points": len(traj_rel),
        "key": key_used,
        "is_slider": duration >= slider_threshold,
    }

def load_existing_data():
    """Wczytuje istniejące dane z pliku"""
    global existing_count
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                existing_count = len(data)
                print(f"📂 Znaleziono istniejący plik z {existing_count} trajektoriami")
                return data
        except Exception as e:
            print(f"⚠️ Błąd wczytywania pliku: {e}")
            print("   Tworzę backup i zaczynam od nowa")
            # Tworzenie backupu uszkodzonego pliku
            try:
                import shutil
                backup_name = f"{output_file}.backup_{int(time.time())}"
                shutil.copy2(output_file, backup_name)
                print(f"   Backup zapisany jako: {backup_name}")
            except:
                pass
            return []
    else:
        print(f"📄 Plik {output_file} nie istnieje - zostanie utworzony")
        return []

def save_dataset():
    """Zapisuje dane DODAJĄC nowe trajektorie do istniejących"""
    global session_saved_count
    
    # Wczytaj istniejące dane
    existing_data = load_existing_data()
    
    # Pobierz nowe trajektorie
    with data_lock:
        new_data = list(new_trajectory_data)
        new_count = len(new_data)
    
    # Jeśli nie ma nowych danych, nie zapisuj
    if new_count == session_saved_count:
        return
    
    # Połącz istniejące z nowymi
    combined_data = existing_data + new_data
    
    # Zapisz połączone dane
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(combined_data, f, ensure_ascii=False)
        
        delta = new_count - session_saved_count
        session_saved_count = new_count
        total = len(combined_data)
        
        print(f"💾 Dodano {delta} nowych trajektorii (sesja: {new_count}, łącznie w pliku: {total})")
    except Exception as e:
        print(f"❌ Błąd zapisu: {e}")

def save_worker():
    while not stop_event.is_set():
        save_event.wait()
        if stop_event.is_set():
            break
        time.sleep(0.03)  # koalescencja wielu szybkich zapisów
        save_event.clear()
        save_dataset()
    if save_event.is_set():
        save_dataset()

def request_save():
    save_event.set()

# --- Ciągłe próbkowanie kursora ---
def sampler():
    enable_high_res_timer()
    try:
        local_batch = []
        last_flush = time.perf_counter()
        next_tick = time.perf_counter()
        while not stop_event.is_set():
            # Odczyt pozycji
            x, y = get_cursor_pos()
            t = time.perf_counter()
            local_batch.append((t, x, y))

            # Flush batchem (rzadziej blokujemy lock)
            if len(local_batch) >= batch_flush_size or (t - last_flush) >= batch_flush_time:
                with samples_lock:
                    samples.extend(local_batch)
                local_batch.clear()
                last_flush = t

            # Precyzyjny sleep
            next_tick += sample_interval
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                # jeśli system nie wyrabia, nadgonimy bez nadmiernego busy-waitu
                next_tick = time.perf_counter()
    finally:
        disable_high_res_timer()

# --- Wycinanie segmentu z bufora ciągłego ---
def extract_segment(t0, t1, pos0=None, pos1=None):
    # Zwraca listę (t_abs, x, y) dla t in [t0, t1] z ewentualnymi punktami granicznymi
    with samples_lock:
        snap = list(samples)

    seg = [(t, x, y) for (t, x, y) in snap if (t0 <= t <= t1)]
    # Dołóż punkt początkowy jeśli brak próbki przy t0
    if pos0 is not None:
        need_start = (not seg) or (seg[0][0] > t0 + sample_interval/2)
        if need_start:
            seg.insert(0, (t0, pos0[0], pos0[1]))
    # Dołóż punkt końcowy jeśli brak próbki przy t1
    if pos1 is not None:
        need_end = (not seg) or (seg[-1][0] < t1 - sample_interval/2)
        if need_end:
            seg.append((t1, pos1[0], pos1[1]))
    return seg

def finalize_segment(t1, pos1):
    global active_key, segment_t0, segment_pos0
    if active_key is None or segment_t0 is None:
        return False
    # Wytnij trajektorię z bufora ciągłego
    seg_abs = extract_segment(segment_t0, t1, pos0=segment_pos0, pos1=pos1)
    traj = normalize_traj(seg_abs, segment_t0)  # [[x,y,t_rel], ...]
    info = calculate_traj_info(traj, active_key)
    with data_lock:
        new_trajectory_data.append({"trajectory": traj, **info})
    
    # Wyświetl info o nowej trajektorii
    slider_info = "🎯 SLIDER" if info["is_slider"] else "➡️ zwykła"
    print(f"  [{slider_info}] Klawisz: {active_key.upper()}, Długość: {info['length']:.1f}px, Czas: {info['duration']:.3f}s, Punkty: {info['points']}")
    
    # Reset segmentu
    active_key = None
    segment_t0 = None
    segment_pos0 = None
    request_save()
    return True

# --- Klawiatura (Z/X segmenty; F12 wyjście) ---
def on_press(key):
    global active_key, segment_t0, segment_pos0
    c = getattr(key, "char", None)
    if c:
        c = c.lower()

    if c in ("z", "x"):
        if active_key is None:
            active_key = c
            segment_t0 = time.perf_counter()
            segment_pos0 = get_cursor_pos()
        elif active_key != c:
            # domknij poprzedni i zacznij nowy
            t1 = time.perf_counter()
            pos1 = get_cursor_pos()
            finalize_segment(t1, pos1)
            active_key = c
            segment_t0 = time.perf_counter()
            segment_pos0 = get_cursor_pos()
        return

    if key == exit_key:
        # domknij, jeśli coś trwa
        if active_key is not None and segment_t0 is not None:
            finalize_segment(time.perf_counter(), get_cursor_pos())
        stop_event.set()
        request_save()

def on_release(key):
    global active_key, segment_t0, segment_pos0
    c = getattr(key, "char", None)
    if c:
        c = c.lower()
    if c in ("z", "x"):
        if active_key == c and segment_t0 is not None:
            t1 = time.perf_counter()
            pos1 = get_cursor_pos()
            finalize_segment(t1, pos1)

def main():
    print("\n" + "="*60)
    print("       REJESTRACJA TRAJEKTORII MYSZY (tryb DODAWANIA)")
    print("="*60)
    
    # Sprawdź istniejący plik
    existing_data = load_existing_data()
    
    print("\n📋 Sterowanie:")
    print("  Z / X  - przytrzymaj, by nagrywać; puść, by zapisać")
    print("  F12    - zakończ program i zapisz")
    print(f"\n⚙️ Parametry:")
    print(f"  • Częstotliwość próbkowania: ~{int(1/sample_interval)} Hz")
    print(f"  • Próg slidera: {slider_threshold}s")
    print(f"  • Plik wyjściowy: {output_file}")
    print("\n🚀 Rozpoczynam nasłuchiwanie...\n")

    sw = threading.Thread(target=save_worker, daemon=True)
    smp = threading.Thread(target=sampler, daemon=True)
    sw.start()
    smp.start()

    kl = keyboard.Listener(on_press=on_press, on_release=on_release)
    kl.start()

    try:
        stop_event.wait()
    except KeyboardInterrupt:
        pass
    finally:
        kl.stop(); kl.join()
        sw.join(); smp.join()
        
        with data_lock:
            new_count = len(new_trajectory_data)
        
        print("\n" + "="*60)
        print(f"✅ Koniec sesji. Dodano {new_count} nowych trajektorii")
        print(f"📁 Dane zapisane w: {output_file}")
        print("="*60 + "\n")

if __name__ == "__main__":
    main()