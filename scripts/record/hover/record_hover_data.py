"""
Nagrywa ruchy myszy dla Imitation Learning.
Hover po liniach tekstu przez 30 min.

Zbiera:
- Timestamp
- Mouse position (x, y)
- Mouse velocity (vx, vy)
- Czy kursor nad tekstem (manual label)
"""

import time
import json
from datetime import datetime
from pathlib import Path
from pynput import mouse
import threading

class MouseRecorder:
    def __init__(self, duration_minutes=30, fps=20):
        self.duration = duration_minutes * 60  # sekundy
        self.fps = fps
        self.interval = 1.0 / fps  # 0.05s dla 20fps
        
        self.recording = []
        self.start_time = None
        self.last_pos = (0, 0)
        self.is_recording = False
        
    def on_move(self, x, y):
        """Callback z pynput - zapisuje pozycję"""
        self.last_pos = (x, y)
    
    def record_loop(self):
        """Główna pętla nagrywania w stałym FPS"""
        self.start_time = time.time()
        frame_count = 0
        
        print(f"🔴 Recording started - {self.duration/60:.0f} minutes")
        print("Move your mouse naturally over text lines...")
        print("Press Ctrl+C to stop early\n")
        
        while self.is_recording:
            current_time = time.time() - self.start_time
            
            if current_time >= self.duration:
                break
                
            # Oblicz velocity
            if len(self.recording) > 0:
                prev = self.recording[-1]
                vx = (self.last_pos[0] - prev['x']) / self.interval
                vy = (self.last_pos[1] - prev['y']) / self.interval
            else:
                vx, vy = 0, 0
            
            # Zapisz frame
            frame = {
                'timestamp': current_time,
                'frame': frame_count,
                'x': self.last_pos[0],
                'y': self.last_pos[1],
                'vx': vx,
                'vy': vy,
            }
            self.recording.append(frame)
            
            # Progress
            if frame_count % (self.fps * 10) == 0:  # Co 10s
                elapsed = current_time / 60
                print(f"⏱️  {elapsed:.1f} / {self.duration/60:.0f} min | Frames: {frame_count}")
            
            frame_count += 1
            time.sleep(self.interval)
        
        print(f"\n✅ Recording complete: {len(self.recording)} frames")
    
    def start(self):
        """Start recording"""
        self.is_recording = True
        
        # Mouse listener w osobnym wątku
        listener = mouse.Listener(on_move=self.on_move)
        listener.start()
        
        # Recording loop
        try:
            self.record_loop()
        except KeyboardInterrupt:
            print("\n⚠️  Recording interrupted by user")
        finally:
            self.is_recording = False
            listener.stop()
    
    def save(self, filename):
        """Zapisz do JSON"""
        output_path = Path('data/recordings') / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'duration': self.duration,
            'fps': self.fps,
            'total_frames': len(self.recording),
            'recorded_at': datetime.now().isoformat(),
        }
        
        data = {
            'metadata': metadata,
            'frames': self.recording
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Saved to: {output_path}")
        print(f"   Frames: {len(self.recording)}")
        print(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    # Test - nagraj 1 minutę zamiast 30
    recorder = MouseRecorder(duration_minutes=1, fps=20)
    recorder.start()
    recorder.save(f"hover_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")