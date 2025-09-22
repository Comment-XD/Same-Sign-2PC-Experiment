import numpy as np
from typing import Optional

class FixedPointEncoder:
    def __init__(self, scale:Optional[int]=1, precision_bits:int=16):
        self._precision_bits = scale * precision_bits
        self._scale = int(2**self._precision_bits)
        
    def encode(self, tensor):
        enc_tensor = np.array(tensor * self._scale, dtype=np.int64)
        return enc_tensor
    
    def decode(self, tensor):
        if self._scale > 1:
            correction = (tensor < 0).astype(np.uint64)
            dividend = np.divide(tensor, self._scale - correction).astype(np.uint64)
            
            remainder = tensor % self._scale
            remainder += (remainder == 0).astype(np.uint64) * self._scale * correction

            tensor = (dividend.astype(np.float64) + remainder.astype(np.float64)) / self._scale
        
        return tensor
        
        