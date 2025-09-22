from .module import Module
from src.rng import generate_random_kbit_tensor
from src.additive_tensor import AdditiveSecretTensor
from src.beaver import BeaverTripleProtocol

class Linear(Module):
    def __init__(self, in_features, out_features, bias=False, share_mode:str="uniform"):
        weights = generate_random_kbit_tensor(size=(in_features, out_features))
        self.in_feat = in_features
        self.out_feat = out_features
        self._weights = AdditiveSecretTensor(weights, share_mode=share_mode)
        
        if bias:
            self.bias = generate_random_kbit_tensor(size=(out_features, 1))
            self.bias = AdditiveSecretTensor(self.bias, share_mode=share_mode)
            
        self.btp = BeaverTripleProtocol()
        
    def forward(self, x):
        x = self.btp(x, self._weights)
        return x
    
    def __repr__(self):
        return f"Linear(in_feat={self.in_feat}, out_feat={self.out_feat})"