"""
This script generates box plots for .npy files whose file names start with "weight_" in the designated "numpy_dir" 
By the implementation of Model Converter, those .npy files dumped with file names starting with "weight_"
represent the weight values of convolution-type (conv2d, dw conv, ...) operations.

The output box plots plot the per-output-channel distributions of the corresponding operations' weight values.
The x-axis of the box plot represnets each output channel, and the y-axis represents the value of these channels' weights.

Note that, for now, the script is only implemented for analyzing weights. Activation tensors are not handled yet.
The analysis should be meant for weights and activation tensors, so this script "should be further modified".

Arguments:
numpy_dir: required. the directory storing numpy files for analyzing.
output_dir: not required. the directory to store the generating box plots.

Outputs:
For each .npy file whose file name starts with "weight_" in numpy_dir, output a box plot figure 
with the same file name to output_dir.
"""


import os
import numpy as np
import matplotlib.pyplot as plt 
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("numpy_dir", 
    help="Directory to load in the numpy files for analyzing.")
parser.add_argument("--output_dir",
    help="Directory to stored the result plots." + 
        "By default, results are stored in the current directory.", 
    default=os.getcwd())

args = parser.parse_args()
params_dir = args.numpy_dir
output_dir = args.output_dir


for param in os.listdir(params_dir):
    file_name = param.replace(".npy", "")
    save_path = os.path.join(output_dir, file_name)
    prefix = file_name.split('_')[:1]
    if prefix[0] == 'weight':
        weight = np.load(os.path.join(params_dir, param))
        per_output_channel_kernel_list = []

        if len(weight.shape) == 4:
            weight = np.reshape(weight, (weight.shape[0], weight.shape[1]*weight.shape[2]*weight.shape[3]))
            fig_x_size = weight.shape[0] * 0.39
            for output_channel_kernel in weight:
                per_output_channel_kernel_list.append(output_channel_kernel)
                
                    
        elif len(weight.shape) == 5:  # group conv2d and dw conv2d
            weight = np.reshape(weight, (weight.shape[0], weight.shape[1], weight.shape[2]*weight.shape[3]*weight.shape[4]))
            output_channel_num = weight.shape[0] * weight.shape[1]  # total output channels after a group conv = groups * output_multiplier
            fig_x_size = output_channel_num * 0.39  # explictly set figure size so that output fig's x-axis won't be so crowded if channel num is too big.
            for kernel_group in weight:
                for output_channel_kernel in kernel_group:  # for dw conv2d there should be output_multiplier kernels in each group.
                    per_output_channel_kernel_list.append(output_channel_kernel)

        plt.figure(figsize=(fig_x_size,8))
        plt.xlabel("output channel index")
        plt.ylabel("range")
        plt.boxplot(per_output_channel_kernel_list)
        plt.savefig(save_path + '.png')
        plt.close()