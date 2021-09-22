"""
This script calculates the percentage of sparsity for .npy files whose file names start with "weight_" in the designated "numpy_dir"
and prints the results in the format:
.npy file name: sparsity percentage 
By the implementation of Model Converter, those .npy files dumped with file names starting with "weight_"
represent the weight values of convolution-type (conv2d, dw conv, ...) operations.

Note that, for now, the script is only implemented for analyzing weights. Activation tensors are not handled yet.
The analysis should be meant for weights and activation tensors, so this script "should be further modified".
"""

import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("numpy_dir", 
    help="Directory to load in the numpy files for analyzing.")

args = parser.parse_args()
params_dir = args.numpy_dir

for param in os.listdir(params_dir):
    prefix = param.split('_')[:1]
    if prefix[0] == 'weight':
        weight = np.load(os.path.join(params_dir, param))
        num_of_zero = weight.size - np.count_nonzero(weight)
        sparsity = round(num_of_zero / weight.size * 100, 2)
        file_name = param.replace(".npy", "")
        print('{:50}:{}%'.format(file_name, sparsity))