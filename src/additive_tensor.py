import numpy as np
from src.rng import generate_additive_shares, generate_same_sign_additive_shares
from typing import Literal

share_methods = {"same_sign": generate_same_sign_additive_shares,
                 "uniform": generate_additive_shares}

class AdditiveSecretTensor(object):
    def __init__(self, 
                 tensor, 
                 ring_size:int=2**64, 
                 share_mode:Literal["same_sign", "uniform"]="uniform") -> None:
        
        share_generator = share_methods[share_mode]
        self._shares = share_generator(tensor, ring_size=ring_size)
        self.size = tensor.shape
    
    @staticmethod
    def from_shares(shares):
        obj = AdditiveSecretTensor.__new__(AdditiveSecretTensor)
        obj._shares = shares
        obj.size = shares[0].shape
        return obj 
    
    def get_plain_text(self):
        return np.sum(self._shares, axis=0).astype(np.int64)
    
    def __add__(self, additive_tensor):
        assert isinstance(additive_tensor, AdditiveSecretTensor), "Operand must be AdditiveSecretTensor"

        result = (self._shares + additive_tensor._shares).astype(np.int64)
        add_tensor = AdditiveSecretTensor.from_shares(result)
        
        return add_tensor

    def __sub__(self, additive_tensor):
        assert isinstance(additive_tensor, AdditiveSecretTensor), "Operand must be AdditiveSecretTensor"

        result = (self._shares - additive_tensor._shares).astype(np.int64)
        add_tensor = AdditiveSecretTensor.from_shares(result)
        
        return add_tensor
    
    def __mul__(self, additive_tensor):
        assert isinstance(additive_tensor, AdditiveSecretTensor), "Operand must be AdditiveSecretTensor"

        result = (self._shares * additive_tensor._shares).astype(np.int64)
        add_tensor = AdditiveSecretTensor.from_shares(result)
        
        return add_tensor
    
    def __matmul__(self, additive_tensor):
        assert isinstance(additive_tensor, AdditiveSecretTensor), "Operand must be AdditiveSecretTensor"

        result = (self._shares @ additive_tensor._shares).astype(np.int64)
        add_tensor = AdditiveSecretTensor.from_shares(result)
        
        return add_tensor

    def __repr__(self):
        return f"AddTensor(shares={self._shares})"