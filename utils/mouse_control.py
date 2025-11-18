"""
Kontrola myszy - wykonywanie ruchów z interpolacją.
Naturalny ruch między punktami zamiast teleportacji.
"""

import time
import numpy as np
from pynput.mouse import Controller, Button
import math

class MouseController:
    def __init__(self, fps=20):
        self.mouse = Controller()
        self.fps = fps
        self.interval = 1.0 / fps  # 0.05s dla 20fps
        
    def get_position(self):
        """Pobierz aktualną pozycję myszy"""
        pos = self.mouse.position
        return np.array([pos[0], pos[1]], dtype=np.float32)
    
    def move_to(self, target_x, target_y, duration=0.3, curve=True):
        """
        Przesuń mysz do punktu docelowego z interpolacją.
        
        Args:
            target_x, target_y: punkt docelowy
            duration: czas ruchu w sekundach
            curve: czy używać krzywej (naturalniejszy ruch)
        """
        start_pos = self.get_position()
        target_pos = np.array([target_x, target_y], dtype=np.float32)
        
        num_steps = int(duration * self.fps)
        if num_steps < 1:
            num_steps = 1
        
        for i in range(num_steps + 1):
            t = i / num_steps
            
            if curve:
                # Bezier curve dla naturalności
                t_curved = self._ease_in_out(t)
            else:
                t_curved = t
            
            # Interpolacja liniowa
            current_pos = start_pos + (target_pos - start_pos) * t_curved
            
            # Dodaj subtelny noise (ludzki tremor)
            noise_x = np.random.normal(0, 0.5)
            noise_y = np.random.normal(0, 0.5)
            
            current_pos[0] += noise_x
            current_pos[1] += noise_y
            
            # Ustaw pozycję
            self.mouse.position = (int(current_pos[0]), int(current_pos[1]))
            
            if i < num_steps:
                time.sleep(self.interval)
    
    def _ease_in_out(self, t):
        """Ease-in-out function dla smooth acceleration/deceleration"""
        return t * t * (3.0 - 2.0 * t)
    
    def click(self, button=Button.left, clicks=1):
        """Kliknij przycisk myszy"""
        for _ in range(clicks):
            self.mouse.click(button)
            time.sleep(0.05)
    
    def execute_trajectory(self, points, duration_per_segment=0.3):
        """
        Wykonaj cały trajektorię (listę punktów).
        
        Args:
            points: lista [(x, y), (x, y), ...]
            duration_per_segment: czas między punktami
        """
        if len(points) < 2:
            return
        
        print(f"🎯 Executing trajectory: {len(points)} points")
        
        for i, (x, y) in enumerate(points):
            print(f"   → Point {i+1}/{len(points)}: ({x:.0f}, {y:.0f})")
            self.move_to(x, y, duration=duration_per_segment)
            time.sleep(0.1)  # Pauza między punktami
        
        print("✅ Trajectory complete")


# Test
if __name__ == "__main__":
    print("🧪 Testing MouseController")
    print("⚠️  Mouse will move in 3 seconds!")
    time.sleep(3)
    
    controller = MouseController(fps=20)
    
    # Test 1: Pojedynczy ruch
    print("\n📍 Current position:", controller.get_position())
    print("Moving to (500, 500)...")
    controller.move_to(500, 500, duration=1.0)
    print("📍 New position:", controller.get_position())
    
    # Test 2: Trajektoria (kwadrat)
    print("\n🔲 Drawing square trajectory...")
    square = [
        (400, 400),
        (600, 400),
        (600, 600),
        (400, 600),
        (400, 400)
    ]
    controller.execute_trajectory(square, duration_per_segment=0.5)
    
    print("\n✅ Test complete")