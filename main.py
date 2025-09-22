from src.additive_tensor import AdditiveSecretTensor

import torchvision
import torchvision.transforms as transforms

from src.rng import *
from src.linear import Linear
from src.conv2d import Conv2d

import logging
import os
from datetime import datetime

def trial(nimages:int=10, verbose:bool=True):
    
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = os.path.join(log_dir, f"trial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        filename=log_filename,
        filemode='w',
        format='%(asctime)s | %(levelname)s | %(message)s',
        level=logging.INFO
    )

    same_sign_conv2d = Conv2d(in_channels=3, out_channels=3, kernel_size=3, share_mode="same_sign")
    uniform_conv2d = Conv2d(in_channels=3, out_channels=3, kernel_size=3)
        
    uniform_conv2d.plain_txt_weights = same_sign_conv2d.plain_txt_weights
    uniform_conv2d._weights = AdditiveSecretTensor(uniform_conv2d.plain_txt_weights)
    
    same_sign_bits, uniform_same_sign_bits = 0, 0
    
    data = torchvision.datasets.CIFAR10("./data", train=True, transform=transforms.ToTensor(), download=True)
    
    for i in range(nimages):
        
        # Images are going to be positive in the beginning
        img = (data[i][0] * 255).long().numpy()
        # print(img)
        img = AdditiveSecretTensor(img, share_mode="same_sign")
        # print(img._shares)
        
        same_sign_result = same_sign_conv2d(img)
        uniform_result = uniform_conv2d(img)
        
        s1, s2 = same_sign_result._shares
        num_same_sign_bits = (np.sign(s1) == np.sign(s2)).sum()
        same_sign_bits += num_same_sign_bits
        
        s1, s2 = uniform_result._shares
        num_uniform_same_sign_bits = (np.sign(s1) == np.sign(s2)).sum()
        uniform_same_sign_bits += num_uniform_same_sign_bits
        
        if verbose and i % (nimages // 10) == 0:
            logging.info(f"T{i}| Uniform: {num_uniform_same_sign_bits}")
            logging.info(f"T{i}| Same Sign: {num_same_sign_bits}")
            print(f"T{i}| Uniform: {num_uniform_same_sign_bits}")
            print(f"T{i}| Same Sign: {num_same_sign_bits}")
    
    mean_same_sign_bits = same_sign_bits / nimages
    mean_uniform_same_sign_bits = uniform_same_sign_bits / nimages
    
    logging.info(f"Total Trials:{nimages} - Mean Same Sign Bits: {mean_same_sign_bits} - Mean Uniform Same Sign Bits: {mean_uniform_same_sign_bits}")
    print(f"Total Trials:{nimages} - Mean Same Sign Bits: {mean_same_sign_bits} - Mean Uniform Same Sign Bits: {mean_uniform_same_sign_bits}")
    
if __name__ == "__main__":
    trial(nimages=10000)