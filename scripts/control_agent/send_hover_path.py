"""
Send hover-bot dots as a polyline path to ControlAgent over UDP.

Usage:
  python scripts/control_agent/send_hover_path.py --json data/processed/<file>.json \
      --port 8765 --speed-factor 1.0 --gap-boost 3.0 --line-jump-boost 1.5 \
      [--offset-x 0 --offset-y 0]

Notes:
  - Assumes dot coordinates are already in screen space; use --offset-* if needed.
  - ControlAgent must be running with UDP receiver (see control_agent.py --port).
"""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_sequences(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected a list of sequences in JSON")
    return data


def _as_rect(box: List[List[float]]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    x0, x1 = int(min(xs)), int(max(xs))
    y0, y1 = int(min(ys)), int(max(ys))
    return (x0, y0, x1, y1)


def _group_lines(seqs: List[Dict[str, Any]]) -> List[List[int]]:
    # grouping by average dot Y (fallback to box center), tolerance by box height
    entries = []  # (idx, min_y, max_y, line_y, height)
    for i, s in enumerate(seqs):
        box = s.get("box") or []
        if not box:
            continue
        ys = [float(p[1]) for p in box]
        min_y, max_y = min(ys), max(ys)
        height = max(1.0, max_y - min_y)
        dots = s.get("dots") or []
        if dots:
            line_y = float(sum(d[1] for d in dots) / len(dots))
        else:
            line_y = 0.5 * (min_y + max_y)
        entries.append((i, min_y, max_y, line_y, height))

    groups: List[List[int]] = []
    ranges: List[Tuple[float, float]] = []
    for i, min_y, max_y, line_y, h in sorted(entries, key=lambda t: t[3]):
        placed = False
        for gi, (gmin, gmax) in enumerate(ranges):
            gc = 0.5 * (gmin + gmax)
            gh = max(1.0, gmax - gmin)
            if abs(line_y - gc) <= 0.45 * max(h, gh):
                ranges[gi] = (min(gmin, min_y), max(gmax, max_y))
                groups[gi].append(i)
                placed = True
                break
        if not placed:
            ranges.append((min_y, max_y))
            groups.append([i])
    return groups

def _inside_any(x: float, y: float, rects: List[Tuple[int,int,int,int]]) -> bool:
    for (x0,y0,x1,y1) in rects:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return True
    return False


def _build_path_and_meta(
    seqs: List[Dict[str, Any]],
    *,
    offset_x: int = 0,
    offset_y: int = 0,
) -> Tuple[List[Dict[str,int]], List[Tuple[int,int,int,int]], List[int]]:
    # Collect gap rects
    gaps: List[Tuple[int,int,int,int]] = []
    for s in seqs:
        if float(s.get("confidence", 0.0)) < 0.0:
            rect = _as_rect([[float(x), float(y)] for x, y in s.get("box", [])])
            gaps.append(rect)

    # Non-gap indices
    real_idx = [i for i, s in enumerate(seqs) if float(s.get("confidence", 0.0)) >= 0.0]
    groups = _group_lines([seqs[i] for i in real_idx])

    points: List[Dict[str,int]] = []
    line_jump_indices: List[int] = []
    seg_index = -1
    for g in groups:
        # order words in line by min_x
        order = sorted(g, key=lambda i: min(p[0] for p in seqs[real_idx[i]].get("box", [[0,0]])))
        first_in_line = True
        for i_local in order:
            s = seqs[real_idx[i_local]]
            dots = s.get("dots") or []
            if not dots:
                continue
            if first_in_line:
                # mark jump from previous line
                if points:
                    line_jump_indices.append(max(0, seg_index))
                first_in_line = False
            for d in dots:
                x, y = int(round(d[0])) + offset_x, int(round(d[1])) + offset_y
                points.append({"x": x, "y": y})
                seg_index += 1
    # Detect TAB-like segments: >=80% of segment length outside any box
    rects_all = [
        _as_rect([[float(x), float(y)] for x, y in s.get("box", [])])
        for s in seqs if float(s.get("confidence", 0.0)) >= 0.0 and s.get("box")
    ]
    def _outside_ratio(p0: Tuple[int,int], p1: Tuple[int,int], samples: int = 9) -> float:
        if samples <= 1:
            return 1.0
        out = 0
        for k in range(samples):
            t = k / (samples - 1)
            x = p0[0] + t * (p1[0] - p0[0])
            y = p0[1] + t * (p1[1] - p0[1])
            if not _inside_any(x, y, rects_all):
                out += 1
        return out / samples
    for i in range(len(points) - 1):
        if _outside_ratio((points[i]["x"], points[i]["y"]), (points[i+1]["x"], points[i+1]["y"])) >= 0.8:
            line_jump_indices.append(i)
    return points, gaps, line_jump_indices


def send_udp(port: int, payload: Dict[str, Any]):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    data = json.dumps(payload).encode("utf-8")
    sock.sendto(data, ("127.0.0.1", port))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, required=True, help="Path to hover_bot JSON with sequences")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--speed-factor", type=float, default=1.0)
    ap.add_argument("--gap-boost", type=float, default=3.0)
    ap.add_argument("--line-jump-boost", type=float, default=1.5)
    ap.add_argument("--min-dt", type=float, default=0.004)
    ap.add_argument("--offset-x", type=int, default=0)
    ap.add_argument("--offset-y", type=int, default=0)
    ap.add_argument("--speed", type=str, default="normal", choices=["slow","normal","fast"])
    ap.add_argument("--min-total-ms", type=float, default=0.0)
    args = ap.parse_args()

    seqs = _load_sequences(Path(args.json))
    points, gaps, line_jumps = _build_path_and_meta(
        seqs, offset_x=args.offset_x, offset_y=args.offset_y
    )
    if len(points) < 2:
        print("[send_hover_path] Not enough points.")
        return

    payload = {
        "cmd": "path",
        "points": points,
        "speed": args.speed,
        "min_total_ms": float(args.min_total_ms),
        "speed_factor": float(args.speed_factor),
        "min_dt": float(args.min_dt),
        "gap_rects": [list(r) for r in gaps],
        "gap_boost": float(args.gap_boost),
        "line_jump_indices": line_jumps,
        "line_jump_boost": float(args.line_jump_boost),
    }
    send_udp(args.port, payload)
    print(f"[send_hover_path] Sent path with {len(points)} points, gaps={len(gaps)}, line_jumps={len(line_jumps)} to port {args.port}")


if __name__ == "__main__":
    main()
