import numpy as np

def generate_additive_shares(tensor, ring_size:int=2**64):
    size = tensor.shape
    s1 = np.random.randint(-(ring_size // 2), 
                           (ring_size // 2) - 1, 
                           size=size, 
                           dtype=np.int64)
    
    s2 = (tensor - s1)
    s2 = np.array(s2, dtype=np.int64)
    
    return np.stack([s1, s2], dtype=np.int64)

def generate_same_sign_additive_shares(tensor, ring_size:int=2**64):
    size = tensor.shape
    mask = np.random.rand(*size)
    
    s1 = (tensor * mask).astype(np.int64)
    s2 = (tensor - s1).astype(np.int64)
    
    return s1, s2

def generate_random_kbit_tensor(size, bit_length:int=16):
    bit_range = int(2 ** bit_length)
    
    random_tensor = np.random.randint(-(bit_range // 2), 
                                       (bit_range // 2) - 1, 
                                        size=size, 
                                        dtype=np.int64)
    return random_tensor

def generate_random_positive_kbit_tensor(size, bit_length:int=16):
    bit_range = int(2 ** bit_length)
    
    random_tensor = np.random.randint(0, 
                                      bit_range - 1, 
                                      size=size, 
                                      dtype=np.int64)
    return random_tensor


