from .module import Module

from src.rng import generate_random_kbit_tensor
from src.additive_tensor import AdditiveSecretTensor
from src.beaver import BeaverTripleProtocol, plain_conv2d

import numpy as np

class Conv2d(Module):
    def __init__(self, 
                 in_channels:int, 
                 out_channels:int, 
                 kernel_size:int,
                 stride:int=1,
                 padding:int=0,
                 bias:bool=False,
                 share_mode:str="uniform") -> None:
    
        self.plain_txt_weights = generate_random_kbit_tensor(size=(out_channels, in_channels, kernel_size, kernel_size), bit_length=64)
        self._weights = AdditiveSecretTensor(self.plain_txt_weights, share_mode=share_mode)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.btp = BeaverTripleProtocol(op="conv2d")
    
    def __call__(self, x):
        assert isinstance(x, AdditiveSecretTensor), "img needs to be a two party system"
        
        # Generates the triplets based on the weights and input size
        A, B, C = self.btp.generate_triplets(x, self._weights)
        
        
        A0, A1 = A._shares
        B0, B1 = B._shares
        C0, C1 = C._shares
        
        X0, X1 = x._shares
        W0, W1 = self._weights._shares

        # compute local masked differences
        d0 = (X0.astype(np.int64) - A0.astype(np.int64)).astype(np.int64)
        d1 = (X1.astype(np.int64) - A1.astype(np.int64)).astype(np.int64)
        e0 = (W0.astype(np.int64) - B0.astype(np.int64)).astype(np.int64)
        e1 = (W1.astype(np.int64) - B1.astype(np.int64)).astype(np.int64)

        # "Open" d and e (in real protocol parties exchange and reconstruct these)
        d_open = (d0 + d1).astype(np.int64)
        e_open = (e0 + e1).astype(np.int64)

        # Compute the full Z (plaintext) using the formula:
        # Z = C + conv(d_open, B_full) + conv(A_full, e_open) + conv(d_open, e_open)
        B_full = (B0 + B1).astype(np.int64)
        A_full = (A0 + A1).astype(np.int64)
        C_full = (C0 + C1).astype(np.int64)

        conv_d_B = plain_conv2d(d_open, B_full, self.padding, self.stride)
        conv_A_e = plain_conv2d(A_full, e_open, self.padding, self.stride)
        conv_d_e = plain_conv2d(d_open, e_open, self.padding, self.stride)
        Z_full = (C_full + conv_d_B + conv_A_e + conv_d_e).astype(np.int64)

        # Randomly split Z_full into two shares and return them (simulation of producing local shares)
        Z = AdditiveSecretTensor(Z_full)
        return Z