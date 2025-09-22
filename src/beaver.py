import numpy as np
from src.rng import generate_random_kbit_tensor
from src.additive_tensor import AdditiveSecretTensor
from typing import Literal

def plain_conv2d(input, weights, padding=0, stride=1):
    # input: (batch_size, in_channels, H, W)
    # weights: (out_channels, in_channels, kH, kW)
    if input.ndim == 3:
        input = np.expand_dims(input, axis=0)
    batch_size, in_channels, H, W = input.shape
    out_channels, _, kH, kW = weights.shape

    padded_input = np.pad(input, ((0,0), (0,0), (padding, padding), (padding, padding)), mode='constant', constant_values=0)
    out_H = (H + 2 * padding - kH) // stride + 1
    out_W = (W + 2 * padding - kW) // stride + 1
    output = np.zeros((batch_size, out_channels, out_H, out_W), dtype=np.int64)

    for b in range(batch_size):
        for oc in range(out_channels):
            for i in range(out_H):
                for j in range(out_W):
                    for ic in range(in_channels):
                        patch = padded_input[b, ic, i*stride:i*stride+kH, j*stride:j*stride+kW]
                        output[b, oc, i, j] += np.sum((patch * weights[oc, ic]).astype(np.int64)).astype(np.int64)
    return output


class BeaverTripleProtocol:
    def __init__(self, 
                 op:Literal["matmul", "conv2d"]="matmul",
                 bit_length:int=64, 
                 ring_size:int=2**64) -> None:
        
        self.op = op
        self.ring_size = ring_size
        self.bit_length = bit_length
        
    def generate_triplets(self, x, y):
        
        a = generate_random_kbit_tensor(x.size, self.bit_length)
        b = generate_random_kbit_tensor(y.size, self.bit_length)
        
        A = AdditiveSecretTensor(a, ring_size=self.ring_size)
        B = AdditiveSecretTensor(b, ring_size=self.ring_size)
        
        if self.op == "matmul":
            c = (a @ b).astype(np.int64)
            C = AdditiveSecretTensor(c, ring_size=self.ring_size)
            
            return A, B, C
            
        if self.op == "conv2d":
            c = plain_conv2d(a, b)
            C = AdditiveSecretTensor(c, ring_size=self.ring_size)
            
            return A, B, C
    
    def __call__(self, X, Y):
        A, B, C = self.generate_triplets(X, Y)
        
        A0, A1 = A._shares
        B0, B1 = B._shares
        C0, C1 = C._shares
        
        E = X - A
        F = Y - B
        
        E = np.sum(E._shares, axis=0).astype(np.int64)
        F = np.sum(F._shares, axis=0).astype(np.int64)
        
        Z0 = (C0 + E @ B0 + A0 @ F + E @F).astype(np.int64)
        Z1 = (C1 + E @ B1 + A1 @ F).astype(np.int64)
        
        return AdditiveSecretTensor.from_shares([Z0, Z1])