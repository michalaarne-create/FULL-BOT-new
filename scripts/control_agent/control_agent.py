# control_agent.py - POPRAWIONA WERSJA ZE SCROLLOWANIEM
import argparse
import json
import math
import os
import platform
import queue
import socket
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple
import random
from pynput.keyboard import Key

# Opcjonalnie NumPy
try:
    import numpy as np
    _NP_OK = True
except Exception:
    _NP_OK = False

# Platform I/O
if platform.system() == "Windows":
    import ctypes
    class POINT(ctypes.Structure): _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
    _user32 = ctypes.windll.user32
    try: _winmm = ctypes.windll.winmm
    except Exception: _winmm = None
    SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN, SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN = 76, 77, 78, 79
    def time_begin_period(ms=1):
        if _winmm:
            try: _winmm.timeBeginPeriod(ms)
            except Exception: pass
    def time_end_period(ms=1):
        if _winmm:
            try: _winmm.timeEndPeriod(ms)
            except Exception: pass
    def get_cursor_pos() -> Tuple[int, int]:
        pt = POINT(); _user32.GetCursorPos(ctypes.byref(pt)); return int(pt.x), int(pt.y)
    def set_cursor_pos_raw(x: int, y: int): _user32.SetCursorPos(int(x), int(y))
    def get_virtual_bounds() -> Tuple[int, int, int, int]:
        left, top, w, h = (_user32.GetSystemMetrics(m) for m in [SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN, SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN])
        if w <= 0 or h <= 0: left, top, w, h = 0, 0, _user32.GetSystemMetrics(0), _user32.GetSystemMetrics(1)
        return left, top, left + w - 1, top + h - 1
    def precise_sleep(seconds: float):
        if seconds <= 0: return
        if seconds < 0.002:
            target = time.perf_counter() + seconds
            while time.perf_counter() < target: pass
        else: time.sleep(seconds)
else:
    from pynput import mouse
    _mc = mouse.Controller()
    def time_begin_period(ms=1): pass
    def time_end_period(ms=1): pass
    def get_cursor_pos() -> Tuple[int, int]: x, y = _mc.position; return int(x), int(y)
    def set_cursor_pos_raw(x: int, y: int): _mc.position = (int(x), int(y))
    def get_virtual_bounds() -> Optional[Tuple[int, int, int, int]]: return None
    def precise_sleep(seconds: float): time.sleep(seconds)

SCREEN_BOUNDS = get_virtual_bounds()

def clamp_to_screen(x: int, y: int, margin: int = 1) -> Tuple[int, int]:
    if SCREEN_BOUNDS is None: return x, y
    l, t, r, b = SCREEN_BOUNDS
    return max(l + margin, min(r - margin, x)), max(t + margin, min(b - margin, y))

def clamp_path(points: List[Tuple[int, int]], margin: int = 1) -> List[Tuple[int, int]]:
    return [clamp_to_screen(x, y, margin) for (x, y) in points]

def _path_length(xy: List[Tuple[float, float]]) -> float:
    if len(xy) < 2: return 0.0
    s = 0.0
    for i in range(len(xy) - 1): s += math.hypot(xy[i + 1][0] - xy[i][0], xy[i + 1][1] - xy[i][1])
    return s

def _curvature_sign_flips(points: List[Tuple[float, float]]) -> int:
    n = len(points)
    if n < 3: return 0
    signs = []
    for i in range(1, n-1):
        x0,y0 = points[i-1]; x1,y1 = points[i]; x2,y2 = points[i+1]
        cross = (x1-x0)*(y2-y1) - (y1-y0)*(x2-x1)
        s = 1 if cross > 1e-6 else (-1 if cross < -1e-6 else 0)
        if s != 0: signs.append(s)
    flips = 0
    for i in range(1, len(signs)):
        if signs[i] != signs[i-1]: flips += 1
    return flips

def moving_average(points: List[Tuple[int,int]], win: int) -> List[Tuple[int,int]]:
    if win <= 1 or len(points) <= 2: return points
    win = max(3, win | 1)
    half = win // 2
    out = []
    for i in range(len(points)):
        xsum, ysum, cnt = 0, 0, 0
        for j in range(max(0, i-half), min(len(points), i+half+1)):
            xsum += points[j][0]; ysum += points[j][1]; cnt += 1
        out.append((int(round(xsum/cnt)), int(round(ysum/cnt))))
    return out

def limit_lateral(points: List[Tuple[int,int]], start: Tuple[int,int], end: Tuple[int,int], max_lat: float) -> List[Tuple[int,int]]:
    sx, sy = start; tx, ty = end
    vx, vy = tx - sx, ty - sy
    L = math.hypot(vx, vy) or 1.0
    ux, uy = vx / L, vy / L
    px, py = -uy, ux
    out = []
    for x, y in points:
        rx, ry = x - sx, y - sy
        lat = rx*px + ry*py
        lat = max(-max_lat, min(max_lat, lat))
        longi = rx*ux + ry*uy
        nx = sx + longi*ux + lat*px
        ny = sy + longi*uy + lat*py
        out.append((int(round(nx)), int(round(ny))))
    out[-1] = (tx, ty)
    return out

class ExactLibrary:
    def __init__(self, data_path: str, slider_threshold: float = 0.05, max_path_length: float = 1900.0):
        self.samples: List[Dict[str, Any]] = []
        self.max_path_length = max_path_length
        self._load(data_path, slider_threshold)
        self._debug_statistics()
    
    def _load(self, data_path: str, slider_threshold: float):
        try:
            with open(data_path, "r", encoding="utf-8") as f: data = json.load(f)
        except Exception as e:
            print(f"[ExactLib] load error: {e}"); return
        
        total_read = 0
        rejected_length = 0
        rejected_norm = 0
        rejected_time = 0
        
        for ex in data:
            total_read += 1
            traj = ex.get("trajectory")
            if not traj or len(traj) < 3: continue
            
            pts = [(float(p[0]), float(p[1])) for p in traj]
            times = [float(p[2]) for p in traj]
            times = [t - times[0] for t in times]
            
            if times[-1] <= 0:
                rejected_time += 1
                continue
            
            start, end = pts[0], pts[-1]
            dx, dy = end[0] - start[0], end[1] - start[1]
            norm = math.hypot(dx, dy)
            
            if norm < 2.0:
                rejected_norm += 1
                continue
            
            path_len = _path_length(pts)
            
            if path_len > self.max_path_length:
                rejected_length += 1
                continue
            
            is_slider_from_json = ex.get("is_slider", None)
            if is_slider_from_json is not None:
                is_slider = bool(is_slider_from_json)
            else:
                is_slider = times[-1] >= slider_threshold
            
            self.samples.append({
                "pts": pts,
                "times": times,
                "theta": math.atan2(dy, dx),
                "norm": norm,
                "path_length": path_len,
                "is_slider": is_slider,
                "straightness": float(norm / max(1e-6, path_len)),
                "flips": int(_curvature_sign_flips(pts)),
            })
        
        print(f"\n[ExactLib] Loaded {len(self.samples)} trajectories")
    
    def _debug_statistics(self):
        if not self.samples:
            print("[DEBUG] No trajectories loaded!")
            return
        
        total_count = len(self.samples)
        slider_count = sum(1 for s in self.samples if s["is_slider"])
        avg_straight_dist = sum(s["norm"] for s in self.samples) / total_count
        avg_path_length = sum(s["path_length"] for s in self.samples) / total_count
        
        print(f"[Stats] Total: {total_count}, Sliders: {slider_count}")
        print(f"[Stats] Avg straight: {avg_straight_dist:.0f}px, Avg path: {avg_path_length:.0f}px")
    
    @staticmethod
    def _angle_diff(a: float, b: float) -> float: 
        return abs((a - b + math.pi) % (2 * math.pi) - math.pi)
    
    def pick_best(self, start: Tuple[int, int], target: Tuple[int, int], want_slider: Optional[bool], curvy: str = "auto", curvy_min: Optional[float] = None) -> Optional[Dict[str, Any]]:
        sx, sy = start; tx, ty = target
        vx, vy = tx - sx, ty - sy
        vnorm = math.hypot(vx, vy)
        if vnorm < 1.0: return None
        vtheta = math.atan2(vy, vx)
        thr = {"less": 0.995, "auto": 0.985, "more": 0.970}.get(curvy, 0.985)
        if curvy_min is not None: thr = float(curvy_min)
        pool = [s for s in self.samples if (want_slider is None or s["is_slider"] == want_slider) and s["straightness"] >= thr]
        if not pool:
            pool = sorted([s for s in self.samples if (want_slider is None or s["is_slider"] == want_slider)], key=lambda s: s["straightness"], reverse=True)
            pool = pool[:min(50, len(pool))]
        
        scale_candidates = []
        for s in pool:
            scale = vnorm / max(1e-6, s["norm"])
            if s.get("flips", 0) <= (3 if scale > 1.5 else 5):
                scale_candidates.append(s)
        if not scale_candidates: scale_candidates = pool

        best, best_score = None, 1e9
        for s in scale_candidates:
            angle_pen = self._angle_diff(vtheta, s["theta"])
            scale_pen = abs(math.log(vnorm / max(1e-6, s["norm"])))
            flips_pen = (0.15 if (vnorm / max(1e-6, s["norm"])) > 1.5 else 0.08) * max(0, s.get("flips", 0) - 1)
            straight_bonus = 1.2 - s["straightness"]
            score = angle_pen + 0.25 * scale_pen + flips_pen + straight_bonus
            if score < best_score: best, best_score = s, score
        return best if best else (max(self.samples, key=lambda s: s["straightness"]) if self.samples else None)
    
    @staticmethod
    def retarget(sample: Dict[str, Any], start_xy: Tuple[int, int], end_xy: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], List[float]]:
        sx, sy = start_xy; tx, ty = end_xy
        pts, times = sample["pts"], sample["times"]
        ex0, ey0 = pts[0]; ex1, ey1 = pts[-1]
        evx, evy = ex1 - ex0, ey1 - ey0
        vtx, vty = tx - sx, ty - sy
        en, tn = math.hypot(evx, evy), math.hypot(vtx, vty)
        if en < 1e-6 or tn < 1e-6: return [(sx, sy), (tx, ty)], [0.0, 0.016]
        scale = tn / en
        phi = math.atan2(vty, vtx) - math.atan2(evy, evx)
        c, s = math.cos(phi), math.sin(phi)
        out_pts: List[Tuple[int, int]] = []
        for (x, y) in pts:
            xr, yr = x - ex0, y - ey0
            xr2, yr2 = scale * (xr * c - yr * s), scale * (xr * s + yr * c)
            out_pts.append((int(round(sx + xr2)), int(round(sy + yr2))))
        out_pts[-1] = (tx, ty)
        t_src, t_tgt = times[-1], max(0.016, times[-1] * scale)
        return out_pts, [(t / t_src) * t_tgt for t in times]

def _cum_arclen(pts: List[Tuple[int,int]]) -> List[float]:
    if len(pts) < 2: return [0.0]*len(pts)
    out=[0.0]
    for i in range(1,len(pts)): out.append(out[-1] + math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1]))
    return out

def _ease_min_jerk(u: float) -> float: u = max(0.0, min(1.0, u)); return u*u*u*(10 - 15*u + 6*u*u)

def _blend_lead_in(pts: List[Tuple[int,int]], times: List[float], lead_px: float = 24.0, lead_ms: float = 80.0) -> Tuple[List[Tuple[int,int]], List[float]]:
    n = len(pts)
    if n < 3 or times[-1] <= 0: return pts, times
    cum = _cum_arclen(pts)
    anchor = max(2, min(next((i for i, d in enumerate(cum) if d >= max(lead_px, 0.03 * cum[-1])), n-1), n-1))
    total_T, lead_T = times[-1], min(lead_ms/1000.0, 0.45 * times[-1])
    P0, P3 = pts[0], pts[anchor]
    v0, v1 = (pts[1][0]-pts[0][0], pts[1][1]-pts[0][1]), (pts[anchor][0]-pts[anchor-1][0], pts[anchor][1]-pts[anchor-1][1])
    def unit(v): L = math.hypot(v[0], v[1]) or 1.0; return (v[0]/L, v[1]/L)
    t0, t1 = unit(v0), unit(v1)
    ctrl_len = max(8.0, min(max(lead_px, 0.03 * cum[-1])*0.6, 0.5*math.hypot(P3[0]-P0[0], P3[1]-P0[1])))
    P1, P2 = (P0[0] + t0[0]*ctrl_len, P0[1] + t0[1]*ctrl_len), (P3[0] - t1[0]*ctrl_len, P3[1] - t1[1]*ctrl_len)
    m = max(8, anchor*3)
    lead_pts, lead_times = [], []
    for i in range(m):
        u = _ease_min_jerk(i/(m-1)); a = 1-u
        x = a**3*P0[0] + 3*a**2*u*P1[0] + 3*a*u**2*P2[0] + u**3*P3[0]
        y = a**3*P0[1] + 3*a**2*u*P1[1] + 3*a*u**2*P2[1] + u**3*P3[1]
        lead_pts.append((int(round(x)), int(round(y)))); lead_times.append(lead_T * (i/(m-1)))
    lead_pts[-1] = P3
    rest_pts, rest_times = pts[anchor+1:], [lead_T + (t - times[anchor])* (times[-1] - lead_T) / max(1e-6, times[-1] - times[anchor]) for t in times[anchor+1:]]
    return lead_pts + rest_pts, lead_times + rest_times

def _ease_slow_last_third(times: List[float], strength: float = 0.9) -> List[float]:
    if len(times) < 2 or times[-1] <= 0: return times
    T = times[-1]
    dt = [times[i+1] - times[i] for i in range(len(times)-1)]
    out_dt = [(d * (1.0 + strength * (((times[i]+times[i+1])*0.5/T - 2/3)*3)**1.2)) if (times[i]+times[i+1])*0.5/T > 2/3 else d for i, d in enumerate(dt)]
    scale = T / sum(out_dt) if sum(out_dt) > 0 else 1.0
    new_times = [0.0]
    for d in out_dt: new_times.append(new_times[-1] + d * scale)
    new_times[-1] = T
    return new_times

class UdpReceiver(threading.Thread):
    def __init__(self, port: int, out_queue: queue.Queue):
        super().__init__(daemon=True)
        self.port, self.out_queue = port, out_queue
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", self.port))
        self.stop_event = threading.Event()
    def run(self):
        print(f"[UDP] Listening on 127.0.0.1:{self.port}")
        self.sock.settimeout(0.5)
        while not self.stop_event.is_set():
            try: data, _ = self.sock.recvfrom(65535)
            except socket.timeout: continue
            try:
                msg = json.loads(data.decode("utf-8"))
                if isinstance(msg, dict): self.out_queue.put(msg)
            except Exception: pass

class InputController:
    def __init__(self):
        from pynput import keyboard, mouse
        self.kb_ctrl, self.ms_ctrl = keyboard.Controller(), mouse.Controller()
        
    def press(self, key_or_button: str):
        from pynput import mouse
        if key_or_button == "mouse": self.ms_ctrl.press(mouse.Button.left)
        else: self.kb_ctrl.press(key_or_button)
        
    def release(self, key_or_button: str):
        from pynput import mouse
        if key_or_button == "mouse": self.ms_ctrl.release(mouse.Button.left)
        else: self.kb_ctrl.release(key_or_button)
    
    def scroll(self, notches: int):
        """Scrolluj o określoną liczbę 'ząbków' (dodatnie = w dół, ujemne = w górę)"""
        if notches == 0:
            return
        # W pynput ujemne = w górę, dodatnie = w dół
        self.ms_ctrl.scroll(0, -notches)

    # MAPOWANIE STRING -> KEY / ZNAK
    def _to_key(self, name: str):
        name = name.lower()
        special = {
            "ctrl": Key.ctrl,
            "control": Key.ctrl,
            "shift": Key.shift,
            "alt": Key.alt,
            "cmd": Key.cmd,
            "win": Key.cmd,
            "enter": Key.enter,
            "tab": Key.tab,
            "esc": Key.esc,
            "escape": Key.esc,
            "space": Key.space,
            "backspace": Key.backspace,
            "delete": Key.delete,
        }
        return special.get(name, name)

    def hotkey(self, *names: str, delay: float = 0.02):
        keys = [self._to_key(n) for n in names]
        for k in keys:
            self.kb_ctrl.press(k)
        time.sleep(delay)
        for k in reversed(keys):
            self.kb_ctrl.release(k)

    def type_text(self, text: str, delay: float = 0.0):
        for ch in text:
            self.kb_ctrl.press(ch)
            self.kb_ctrl.release(ch)
            if delay > 0:
                time.sleep(delay)

    def paste(self):
        self.hotkey("ctrl", "v")

    def copy(self):
        self.hotkey("ctrl", "c")

class ControlAgent:
    def __init__(self, cfg_path: str, udp_port: int = 8765, verbose: bool = False):
        self.cfg = self._load_cfg(cfg_path)
        data_rel = self.cfg.get("dataset", {}).get("source", "trajectory.json")
        data_path = data_rel if os.path.isabs(data_rel) else os.path.join(os.path.dirname(os.path.abspath(cfg_path)), data_rel)
        self.lib = ExactLibrary(data_path=data_path, slider_threshold=0.05, max_path_length=1900.0)
        self.cmd_queue: queue.Queue = queue.Queue()
        self.receiver = UdpReceiver(udp_port, self.cmd_queue)
        self.input_ctrl = InputController()
        self.verbose = bool(verbose or os.environ.get("CONTROL_AGENT_VERBOSE") == "1")
        
        print("[Agent] Using OSU trajectories. Movements will be accelerated.")
        if SCREEN_BOUNDS: 
            print(f"[Agent] Screen bounds detected: {SCREEN_BOUNDS}")

    def _load_cfg(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f: return json.load(f)
        except Exception: return {}

    def start(self):
        self.receiver.start()
        print(f"[Agent] Ready. Waiting for UDP commands on port {self.receiver.port}...")
        try:
            while True:
                try: cmd = self.cmd_queue.get(timeout=0.1)
                except queue.Empty: continue
                self.handle_command(cmd)
        except KeyboardInterrupt: pass
        finally:
            self.receiver.stop_event.set()
            self.receiver.join(timeout=1.0)
            print("[Agent] Stopped.")

    def handle_command(self, cmd: Dict[str, Any]):
        c = (cmd.get("cmd") or "move").lower()
        if c == "move":
            self._cmd_move(cmd)
        elif c == "scroll":
            self._cmd_scroll(cmd)
        elif c == "path":
            self._cmd_path(cmd)
        elif c == "keys":
            self._cmd_keys(cmd)
        elif c == "type":
            self._cmd_type(cmd)
        elif c == "paste":
            self.input_ctrl.paste()
        else:
            print(f"[Agent] Unknown command: {c}")

    def _cmd_scroll(self, cmd: Dict[str, Any]):
        """Obsługa komendy scrollowania"""
        direction = cmd.get("direction", "down").lower()
        amount = int(cmd.get("amount", 3))  # liczba notches
        duration = float(cmd.get("duration", 0.5))
        
        print(f"[Scroll] {direction} by {amount} notches")
        
        # Określ kierunek
        if direction == "up":
            amount = -abs(amount)
        else:
            amount = abs(amount)
        
        # Scrolluj
        self._smooth_scroll_notches(amount, duration)

    def _cmd_keys(self, cmd: Dict[str, Any]):
        combo = (cmd.get("combo") or "").lower()
        if not combo:
            return
        parts = [p.strip() for p in combo.split("+") if p.strip()]
        if not parts:
            return
        if self.verbose:
            try:
                print(f"[Keys] combo={'+'.join(parts)}")
            except Exception:
                pass
        if len(parts) == 1:
            key = self.input_ctrl._to_key(parts[0])
            self.input_ctrl.kb_ctrl.press(key)
            time.sleep(0.02)
            self.input_ctrl.kb_ctrl.release(key)
        else:
            self.input_ctrl.hotkey(*parts)

    def _cmd_type(self, cmd: Dict[str, Any]):
        text = cmd.get("text") or ""
        if not text:
            return
        delay = float(cmd.get("delay", 0.0))
        if self.verbose:
            preview = text.replace("\n", " ")
            if len(preview) > 60:
                preview = preview[:57] + "..."
            try:
                print(f"[Type] len={len(text)} delay={delay}s text=\"{preview}\"")
            except Exception:
                pass
        self.input_ctrl.type_text(text, delay=delay)

    def _smooth_scroll_notches(self, total_notches: int, duration: float = 0.5):
        """Scrolluj płynnie o określoną liczbę notches"""
        if total_notches == 0:
            return
            
        abs_notches = abs(total_notches)
        
        if abs_notches <= 3:
            # Mało - scrolluj na raz
            self.input_ctrl.scroll(total_notches)
            time.sleep(0.1)
        else:
            # Dużo - podziel na kroki
            steps = min(abs_notches, 10)  # Max 10 kroków
            notches_per_step = total_notches / steps
            delay_per_step = duration / steps
            
            for i in range(steps):
                # Losowa wariacja dla bardziej ludzkiego ruchu
                this_step = int(notches_per_step + random.uniform(-0.3, 0.3))
                if this_step == 0:
                    this_step = 1 if total_notches > 0 else -1
                    
                self.input_ctrl.scroll(this_step)
                
                # Losowe opóźnienie
                delay = delay_per_step * random.uniform(0.8, 1.2)
                time.sleep(delay)

    def _target_duration_ms(self, dist_px: float, speed_str: str, duration_ms: Optional[float], min_total_ms: float) -> float:
        if duration_ms is not None:
            base = float(duration_ms)
        else:
            d = float(max(0.0, dist_px))
            # Wartości podzielone przez ~2, żeby przyspieszyć ruch
            if d <= 150: base = 375.0 + (d / 150.0) * 125.0    # 375-500ms
            elif d <= 400: base = 500.0 + ((d-150.0)/250.0)*200.0 # 500-700ms
            else: base = 700.0 + min(250.0, (d-400.0)*0.4)     # 700-950ms
        
        base *= random.uniform(0.97, 1.03)
        speed_str = (speed_str or "normal").lower()
        if speed_str == "fast": base *= 0.8
        elif speed_str == "slow": base *= 1.2
        
        return max(max(100.0, float(min_total_ms or 0.0)), base)

    def _densify_linear(self, pts: List[Tuple[int,int]], times: List[float], fps: int = 144) -> Tuple[List[Tuple[int,int]], List[float]]:
        if len(pts) < 2 or times[-1] <= 0: return pts, times
        T, N = times[-1], max(60, int(times[-1] * fps))
        new_times = [i * (T / (N - 1)) for i in range(N)]
        new_pts = []
        j = 0
        for tn in new_times:
            while j + 1 < len(times) and times[j + 1] < tn: j += 1
            if j + 1 >= len(times): new_pts.append(pts[-1]); continue
            t0, t1 = times[j], times[j+1]
            if t1 <= t0: new_pts.append(pts[j]); continue
            a = (tn - t0) / (t1 - t0)
            x = int(round(pts[j][0] + a * (pts[j+1][0] - pts[j][0])))
            y = int(round(pts[j][1] + a * (pts[j+1][1] - pts[j][1])))
            new_pts.append((x, y))
        new_pts[-1] = pts[-1]
        return new_pts, new_times

    def _postprocess_path(self, pts: List[Tuple[int,int]], times: List[float], start: Tuple[int,int], end: Tuple[int,int]) -> Tuple[List[Tuple[int,int]], List[float]]:
        dist = math.hypot(end[0]-start[0], end[1]-start[1])
        win = 5 if dist < 250 else (7 if dist < 600 else 9)
        pts = moving_average(pts, win)
        max_lat = min(40.0, 0.08 * dist)
        pts = limit_lateral(pts, start, end, max_lat)
        pts, times = _blend_lead_in(pts, times, lead_px=24.0, lead_ms=80.0)
        times = _ease_slow_last_third(times, strength=0.95)
        pts, times = self._densify_linear(pts, times, fps=144)
        pts = clamp_path(pts, margin=1)
        pts[-1] = end
        return pts, times

    def _run_timed_points(self, points: List[Tuple[int, int]], times_sec: List[float]):
        if not points or len(points) < 2: return
        time_begin_period(1)
        try:
            t0 = time.perf_counter()
            for i in range(len(points)):
                set_cursor_pos_raw(points[i][0], points[i][1])
                if i < len(points) - 1:
                    delay = (t0 + times_sec[i+1]) - time.perf_counter()
                    if delay > 0: precise_sleep(delay)
        finally: time_end_period(1)

    # --- PATH (polyline) execution helpers ---
    @staticmethod
    def _seg_intersects_rect(p0: Tuple[int,int], p1: Tuple[int,int], rect: Tuple[int,int,int,int]) -> bool:
        x0,y0 = p0; x1,y1 = p1
        rx0,ry0,rx1,ry1 = rect
        if rx0 > rx1: rx0,rx1 = rx1,rx0
        if ry0 > ry1: ry0,ry1 = ry1,ry0
        if max(x0,x1) < rx0 or min(x0,x1) > rx1 or max(y0,y1) < ry0 or min(y0,y1) > ry1:
            return False
        if rx0 <= x0 <= rx1 and ry0 <= y0 <= ry1: return True
        if rx0 <= x1 <= rx1 and ry0 <= y1 <= ry1: return True
        def _ccw(ax,ay,bx,by,cx,cy): return (cy-ay)*(bx-ax) > (by-ay)*(cx-ax)
        def _inter(a,b,c,d):
            (ax,ay),(bx,by),(cx,cy),(dx,dy) = a,b,c,d
            return _ccw(ax,ay,cx,cy,dx,dy) != _ccw(bx,by,cx,cy,dx,dy) and _ccw(ax,ay,bx,by,cx,cy) != _ccw(ax,ay,bx,by,dx,dy)
        A=(x0,y0); B=(x1,y1)
        edges=[((rx0,ry0),(rx1,ry0)),((rx1,ry0),(rx1,ry1)),((rx1,ry1),(rx0,ry1)),((rx0,ry1),(rx0,ry0))]
        return any(_inter(A,B,e0,e1) for (e0,e1) in edges)

    def _build_times_for_path(
        self,
        pts: List[Tuple[int,int]],
        *,
        speed: str = "normal",
        duration_ms: Optional[float] = None,
        min_total_ms: float = 0.0,
        global_speed_factor: float = 1.0,
        min_dt: float = 0.004,
        gap_rects: Optional[List[Tuple[int,int,int,int]]] = None,
        gap_boost: float = 1.0,
        line_jump_indices: Optional[List[int]] = None,
        line_jump_boost: float = 1.0,
    ) -> List[float]:
        if len(pts) < 2:
            return [0.0]
        total_dist = 0.0
        seg_len: List[float] = []
        for i in range(len(pts)-1):
            d = math.hypot(pts[i+1][0]-pts[i][0], pts[i+1][1]-pts[i][1])
            seg_len.append(d); total_dist += d
        if total_dist <= 0.0:
            return [0.0, 0.016]
        base_T = self._target_duration_ms(total_dist, speed, duration_ms, min_total_ms) / 1000.0
        base_T = max(0.03, base_T)
        # global speed applied per-segment as seg_speed
        times = [0.0]
        gap_rects = gap_rects or []
        line_jump_set = set(line_jump_indices or [])
        t_acc = 0.0
        seg_speed = max(0.05, float(global_speed_factor))
        for i, d in enumerate(seg_len):
            share = d / total_dist
            dt = base_T * share
            if any(self._seg_intersects_rect(pts[i], pts[i+1], r) for r in gap_rects):
                dt /= max(1.0, gap_boost)
            if i in line_jump_set:
                dt /= max(1.0, line_jump_boost)
            # per-segment relative speed randomization
            x = round(random.uniform(-5.0, 5.0), 4)
            if random.random() < 0.02:
                x = round(x * 5.0, 4)
            seg_speed = max(0.05, seg_speed * (1.0 + x / 100.0))
            dt /= seg_speed
            dt = max(min_dt, dt)
            t_acc += dt
            times.append(t_acc)
        return times

    def _cmd_path(self, cmd: Dict[str, Any]):
        # Execute polyline path with optional boosts and global speed factor
        pts_in = cmd.get("points") or []
        if not pts_in or len(pts_in) < 2:
            print("[Path] ignored: too few points")
            return
        pts: List[Tuple[int,int]] = [(int(p["x"]), int(p["y"])) for p in pts_in]
        pts = clamp_path(pts, margin=1)
        speed = str(cmd.get("speed", "normal"))
        duration_ms = cmd.get("duration_ms")
        min_total_ms = float(cmd.get("min_total_ms", 0.0))
        speed_factor = float(cmd.get("speed_factor", 1.0))
        min_dt = float(cmd.get("min_dt", 0.004))
        gap_rects = [tuple(map(int, r)) for r in (cmd.get("gap_rects") or [])]
        gap_boost = float(cmd.get("gap_boost", 1.0))
        line_jump_indices = [int(i) for i in (cmd.get("line_jump_indices") or [])]
        line_jump_boost = float(cmd.get("line_jump_boost", 1.0))

        times = self._build_times_for_path(
            pts,
            speed=speed,
            duration_ms=duration_ms,
            min_total_ms=min_total_ms,
            global_speed_factor=speed_factor,
            min_dt=min_dt,
            gap_rects=gap_rects,
            gap_boost=gap_boost,
            line_jump_indices=line_jump_indices,
            line_jump_boost=line_jump_boost,
        )
        pts_d, times_d = self._densify_linear(pts, times, fps=144)
        if self.verbose:
            try:
                print(f"[Path] steps={len(pts_d)} duration={times_d[-1]:.3f}s from=({pts_d[0][0]},{pts_d[0][1]}) to=({pts_d[-1][0]},{pts_d[-1][1]})")
            except Exception:
                pass
        self._run_timed_points(pts_d, times_d)

    def _cmd_move(self, cmd: Dict[str, Any]):
        """Obsługa komendy ruchu - UPROSZCZONE BEZ AUTO-SCROLLOWANIA"""
        tx, ty = int(cmd.get("x", 0)), int(cmd.get("y", 0))
        press = str(cmd.get("press", "none")).lower()
        speed = str(cmd.get("speed", "normal")).lower()
        duration_ms = cmd.get("duration_ms", None)
        curvy = str(cmd.get("curvy", "auto")).lower()
        min_total_ms = float(cmd.get("min_total_ms", 0.0))
        
        # NOWE: Explicite scrollowanie tylko jeśli bot tego zażąda
        needs_scroll = cmd.get("needs_scroll", False)
        scroll_direction = cmd.get("scroll_direction", "down")
        scroll_amount = int(cmd.get("scroll_amount", 3))
        
        sx, sy = get_cursor_pos()
        if self.verbose:
            desc = cmd.get("desc") or cmd.get("label") or cmd.get("text") or cmd.get("role")
            target_info = f" '{desc}'" if desc else ""
            print(f"[Move] to({tx},{ty}) from({sx},{sy}) press={press} speed={speed} curvy={curvy}{target_info}")
        
        # Scrolluj TYLKO jeśli bot explicite tego zażąda
        if needs_scroll:
            print(f"[Agent] Bot requested scroll {scroll_direction} by {scroll_amount} notches")
            
            # Przesuń mysz w bezpieczne miejsce do scrollowania
            if SCREEN_BOUNDS:
                _, _, right, bottom = SCREEN_BOUNDS
                scroll_x = right // 2
                scroll_y = bottom // 2
            else:
                scroll_x = 960
                scroll_y = 540
                
            # Szybki ruch do miejsca scrollowania
            set_cursor_pos_raw(scroll_x, scroll_y)
            time.sleep(0.1)
            
            # Scrolluj
            if scroll_direction == "up":
                scroll_amount = -abs(scroll_amount)
            else:
                scroll_amount = abs(scroll_amount)
                
            self._smooth_scroll_notches(scroll_amount, duration=0.5)
            
            # Poczekaj aż strona się ustabilizuje
            time.sleep(0.3)
            
            # Pobierz nową pozycję kursora
            sx, sy = get_cursor_pos()
        
        # Kontynuuj normalny ruch do celu
        tx, ty = clamp_to_screen(tx, ty)
        dist = math.hypot(tx - sx, ty - sy)

        want_slider: Optional[bool] = True if str(cmd.get("action", "auto")).lower() == "slider" else False
        sample = self.lib.pick_best((sx, sy), (tx, ty), want_slider, curvy=curvy)
        
        if sample is None:
            Tms = self._target_duration_ms(dist, speed, duration_ms, min_total_ms)
            pts = [(sx, sy), (tx, ty)]
            times = [0.0, Tms / 1000.0]
            pts, times = self._densify_linear(pts, times)
            if self.verbose:
                try:
                    print(f"[Path] linear steps={len(pts)} duration={times[-1]:.3f}s from=({pts[0][0]},{pts[0][1]}) to=({pts[-1][0]},{pts[-1][1]})")
                except Exception:
                    pass
            self._run_timed_points(pts, times)
        else:
            pts_raw, times_raw = ExactLibrary.retarget(sample, (sx, sy), (tx, ty))
            T_target_ms = self._target_duration_ms(dist, speed, duration_ms, min_total_ms)
            scale_t = (T_target_ms / 1000.0) / max(1e-6, times_raw[-1])
            times_raw = [t * scale_t for t in times_raw]
            pts_final, times_final = self._postprocess_path(pts_raw, times_raw, (sx, sy), (tx, ty))
            if self.verbose:
                try:
                    print(
                        "[Pick] slider=" + str(sample.get("is_slider")) +
                        f" straight={sample.get('straightness', 0.0):.3f} flips={int(sample.get('flips', 0))}"
                    )
                    print(f"[Path] steps={len(pts_final)} duration={times_final[-1]:.3f}s from=({pts_final[0][0]},{pts_final[0][1]}) to=({pts_final[-1][0]},{pts_final[-1][1]})")
                except Exception:
                    pass
            self._run_timed_points(pts_final, times_final)

        if press == "mouse":
            time.sleep(random.uniform(0.01, 0.03))
            if self.verbose:
                cx, cy = get_cursor_pos()
                print(f"[Click] left at ({cx},{cy})")
            self.input_ctrl.press("mouse")
            time.sleep(random.uniform(0.02, 0.05))
            self.input_ctrl.release("mouse")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Config file (e.g. train.json)")
    parser.add_argument("--port", type=int, default=8765, help="UDP port to listen on")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging of actions and click coordinates")
    args = parser.parse_args()

    try:
        from pynput import keyboard, mouse
        kb_ctrl = keyboard.Controller()
        ms_ctrl = mouse.Controller()
    except Exception as e:
        print("\n[ERROR] Cannot initialize Pynput.")
        print("Make sure the script has proper permissions (Accessibility/Input Monitoring on macOS).")
        print(f"Details: {e}\n")
        return

    agent = ControlAgent(cfg_path=args.config, udp_port=args.port, verbose=args.verbose)
    agent.start()

if __name__ == "__main__":
    main()
