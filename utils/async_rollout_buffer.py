"""
Async Rollout Buffer with CUDA streams for parallel processing
"""

import torch
import numpy as np
from typing import Dict, Generator, Optional
from stable_baselines3.common.buffers import RolloutBuffer


class AsyncRolloutBuffer(RolloutBuffer):
    """
    RolloutBuffer with async GPU transfers using CUDA streams.
    """
    
    def __init__(self, *args, use_cuda_streams: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.use_cuda_streams = use_cuda_streams
        
        if use_cuda_streams and torch.cuda.is_available():
            # Create dedicated CUDA stream for data transfer
            self.transfer_stream = torch.cuda.Stream()
            print("✅ Async Buffer: CUDA streams enabled")
        else:
            self.transfer_stream = None
    
    def get(self, batch_size: Optional[int] = None) -> Generator:
        """
        Get batches with async GPU transfer.
        While GPU processes batch N, CPU prepares batch N+1.
        """
        assert self.full, "Rollout buffer must be full"
        
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        
        # Prepare all batches first (on CPU)
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs
        
        start_idx = 0
        
        # Prefetch first batch
        next_batch = None
        if self.transfer_stream is not None:
            with torch.cuda.stream(self.transfer_stream):
                next_batch = self._prepare_batch(indices[0:batch_size])
        
        while start_idx < len(indices):
            # Current batch indices
            batch_inds = indices[start_idx:start_idx + batch_size]
            
            # Use prefetched batch or prepare now
            if next_batch is not None:
                # Wait for prefetch to complete
                torch.cuda.current_stream().wait_stream(self.transfer_stream)
                batch = next_batch
            else:
                batch = self._prepare_batch(batch_inds)
            
            # Prefetch NEXT batch while GPU processes current
            next_start = start_idx + batch_size
            if next_start < len(indices) and self.transfer_stream is not None:
                next_batch_inds = indices[next_start:next_start + batch_size]
                with torch.cuda.stream(self.transfer_stream):
                    next_batch = self._prepare_batch(next_batch_inds)
            else:
                next_batch = None
            
            # Yield current batch (GPU can process while we prefetch next)
            yield batch
            
            start_idx += batch_size
    
    def _prepare_batch(self, indices):
        """Prepare batch and move to GPU."""
        # Get data
        obs = self._get_samples(indices)
        
        # Move to GPU with non_blocking=True
        if self.device.type == 'cuda':
            for key in obs.__dict__:
                value = getattr(obs, key)
                if isinstance(value, torch.Tensor):
                    setattr(obs, key, value.to(self.device, non_blocking=True))
        
        return obs