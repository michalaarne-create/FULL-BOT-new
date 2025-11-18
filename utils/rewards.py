"""
Funkcje nagród dla różnych behaviors.
"""

import numpy as np

class HoverRewardCalculator:
    """Obliczanie nagród dla hover behavior"""
    
    def __init__(self):
        self.ideal_angle = 15  # stopnie
        self.ideal_y_range = (20, 40)  # px
        
    def calculate(self, new_pos, prev_pos, target_line):
        """
        Args:
            new_pos: [x, y] nowa pozycja
            prev_pos: [x, y] poprzednia pozycja
            target_line: dict {'x1', 'y1', 'x2', 'y2'}
        
        Returns:
            reward: float
            info: dict z breakdown
        """
        reward = 0.0
        info = {}
        
        # Oblicz offset
        x_offset = new_pos[0] - prev_pos[0]
        y_offset = new_pos[1] - prev_pos[1]
        
        # 1. Y_delta reward
        y_reward = self._y_delta_reward(y_offset)
        reward += y_reward
        info['y_delta_reward'] = y_reward
        
        # 2. Angle reward
        angle_reward = self._angle_reward(x_offset, y_offset)
        reward += angle_reward
        info['angle_reward'] = angle_reward
        
        # 3. Distance from line penalty
        dist_reward = self._distance_penalty(new_pos, target_line)
        reward += dist_reward
        info['distance_penalty'] = dist_reward
        
        # 4. Reading flow bonus
        flow_reward = 0.1 if x_offset > 0 else 0
        reward += flow_reward
        info['flow_reward'] = flow_reward
        
        info['total_reward'] = reward
        return reward, info
    
    def _y_delta_reward(self, y_offset):
        """Reward za ruch Y w zakresie [-15, 80]px"""
        if -15 <= y_offset <= 80:
            if self.ideal_y_range[0] <= y_offset <= self.ideal_y_range[1]:
                return 1.0
            elif 0 <= y_offset < self.ideal_y_range[0] or \
                 self.ideal_y_range[1] < y_offset <= 60:
                return 0.5
            else:
                return 0.2
        else:
            return -0.5
    
    def _angle_reward(self, x_offset, y_offset):
        """Reward za naturalny kąt ~15°"""
        if abs(x_offset) < 5:
            return 0
        
        angle = np.abs(np.arctan(y_offset / x_offset) * 180 / np.pi)
        angle_diff = abs(angle - self.ideal_angle)
        
        if angle_diff < 5:
            return 0.5
        elif angle_diff < 15:
            return 0.2
        else:
            return 0
    
    def _distance_penalty(self, pos, line):
        """Kara za odległość od linii"""
        dist_y = abs(pos[1] - line['y1'])
        center_x = (line['x1'] + line['x2']) / 2
        dist_x = abs(pos[0] - center_x)
        
        penalty = 0
        
        if dist_y > 50:
            penalty -= 1.0
        elif dist_y > 25:
            penalty -= 0.3
        
        if dist_x > 50:
            penalty -= 0.5
        
        return penalty


# Test
if __name__ == "__main__":
    calc = HoverRewardCalculator()
    
    line = {'x1': 100, 'y1': 200, 'x2': 800, 'y2': 200}
    prev_pos = np.array([150, 180])
    new_pos = np.array([200, 210])  # +50x, +30y
    
    reward, info = calc.calculate(new_pos, prev_pos, line)
    
    print("🧪 Testing HoverRewardCalculator")
    print(f"Previous pos: {prev_pos}")
    print(f"New pos: {new_pos}")
    print(f"Offset: [{new_pos[0]-prev_pos[0]}, {new_pos[1]-prev_pos[1]}]")
    print(f"\nReward breakdown:")
    for key, value in info.items():
        print(f"  {key}: {value:.2f}")
    
    print("\n✅ Test complete")