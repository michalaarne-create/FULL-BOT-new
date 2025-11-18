"""
Przetwarzanie nagranych danych IL do formatu do BC training.

Input: data/recordings/*.json (z 01_record_data.py)
Output: data/processed/hover_dataset.npz
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

def load_recording(filepath):
    """Wczytaj plik nagrania"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def extract_trajectories(recording, min_length=10):
    """
    Wyciągnij trajektorie (ciągłe segmenty ruchu).
    
    Segment kończy się gdy:
    - Mysz stoi w miejscu > 0.5s
    - Bardzo duży skok (> 200px)
    """
    frames = recording['frames']
    trajectories = []
    current_traj = []
    
    prev_x, prev_y = frames[0]['x'], frames[0]['y']
    still_counter = 0
    
    for frame in frames:
        x, y = frame['x'], frame['y']
        
        # Check if mouse stopped
        dist = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
        
        if dist < 2:  # Praktycznie w miejscu
            still_counter += 1
            if still_counter > 10:  # 0.5s @ 20fps
                # Zakończ trajektorię
                if len(current_traj) >= min_length:
                    trajectories.append(current_traj)
                current_traj = []
                still_counter = 0
        else:
            still_counter = 0
            
            # Check for huge jump (teleportacja)
            if dist > 200:
                if len(current_traj) >= min_length:
                    trajectories.append(current_traj)
                current_traj = []
            
            current_traj.append({
                'x': x,
                'y': y,
                'vx': frame['vx'],
                'vy': frame['vy'],
                'timestamp': frame['timestamp']
            })
        
        prev_x, prev_y = x, y
    
    # Dodaj ostatnią trajektorię
    if len(current_traj) >= min_length:
        trajectories.append(current_traj)
    
    return trajectories

def create_bc_dataset(trajectories, screen_w=1920, screen_h=1080):
    """
    Stwórz dataset dla BC (Behavior Cloning).
    
    Format:
    - observations: [current_x, current_y] (normalized)
    - actions: [next_x, next_y] (normalized)
    """
    observations = []
    actions = []
    
    for traj in trajectories:
        for i in range(len(traj) - 1):
            current = traj[i]
            next_point = traj[i + 1]
            
            # Normalize to [-1, 1]
            obs = np.array([
                current['x'] / screen_w * 2 - 1,
                current['y'] / screen_h * 2 - 1
            ], dtype=np.float32)
            
            action = np.array([
                next_point['x'] / screen_w * 2 - 1,
                next_point['y'] / screen_h * 2 - 1
            ], dtype=np.float32)
            
            observations.append(obs)
            actions.append(action)
    
    return np.array(observations), np.array(actions)

def process_all_recordings(recordings_dir='data/recordings', output_file='data/processed/hover_dataset.npz'):
    """Przetwórz wszystkie nagrania"""
    recordings_path = Path(recordings_dir)
    
    if not recordings_path.exists():
        print(f"❌ Directory not found: {recordings_dir}")
        return
    
    recording_files = list(recordings_path.glob('*.json'))
    
    if len(recording_files) == 0:
        print(f"⚠️  No recordings found in {recordings_dir}")
        return
    
    print(f"📁 Found {len(recording_files)} recording(s)")
    
    all_trajectories = []
    
    for rec_file in tqdm(recording_files, desc="Loading recordings"):
        recording = load_recording(rec_file)
        trajectories = extract_trajectories(recording)
        all_trajectories.extend(trajectories)
    
    print(f"✅ Extracted {len(all_trajectories)} trajectories")
    
    # Create BC dataset
    print("🔄 Creating BC dataset...")
    observations, actions = create_bc_dataset(all_trajectories)
    
    print(f"   Observations: {observations.shape}")
    print(f"   Actions: {actions.shape}")
    
    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_path,
        observations=observations,
        actions=actions,
        num_trajectories=len(all_trajectories)
    )
    
    print(f"💾 Saved to: {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")
    
    # Stats
    print(f"\n📊 Dataset statistics:")
    print(f"   Total samples: {len(observations)}")
    print(f"   Trajectories: {len(all_trajectories)}")
    print(f"   Avg trajectory length: {len(observations) / len(all_trajectories):.1f}")
    
    return output_path


if __name__ == "__main__":
    print("🔄 Processing IL recordings...\n")
    process_all_recordings()
    print("\n✅ Processing complete")