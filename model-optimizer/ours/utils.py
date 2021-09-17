import os
import numpy as np
from mo.graph.graph import Node

# [Eason] dir or file naming cannot contain invalid char
def valid_dir_or_file_name(name_string):
    invalid_chars_in_name = ['/', ':', '*', '?', '<', '>']

    for invalid_char in invalid_chars_in_name:
        if invalid_char in name_string:
            name_string = '_'.join(name_string.split(invalid_char))

    return name_string


def npy_file_naming(node: Node):
    '''
    we observe that in some tflite model, some tensors' names are the same,
    so now we don't use tensor name in the naming of the corresponding dumped numy file to avoid collision.
    the naming rule now is:  ('weight') + layer_id + input_idx + shape
    '''
    ops_that_have_weights = {'Convolution', 'GroupConvolution', 'MatMul', 'ConvolutionBackpropData', 'GroupConvolutionBackpropData'}
    des_port_idx = node.out_port(0).get_destination().idx
    op_node = node.out_port(0).get_destination().node  # op node that comsumes this param
    npy_file_name = 'layer_{}_input_{}_shape_{}'.format(str(op_node.id), str(des_port_idx), get_shape_string(node.value.shape))
    if op_node.type in ops_that_have_weights and des_port_idx == 1: 
        npy_file_name = 'weight_' + npy_file_name

    return npy_file_name


def check_dumped_params_answer(graph, bin_file, dump_numpy_dir):
    for node in graph.get_op_nodes(op='Const'):
        npy_file_name = npy_file_naming(node)
        offset = node['offset']
        size = node['size']
        npy_file = os.path.join(dump_numpy_dir, npy_file_name + '.npy')
        npy_value = np.load(npy_file)
        bin_value = np.fromfile(bin_file, dtype=node.value.dtype, offset=offset, count=size // node.value.dtype.itemsize)
        assert np.array_equal(npy_value.flatten(), bin_value), "Const {}'s .npy file value mismatch with its correspondence in .bin file".format(node.name)

    
def get_shape_string(shape):
    shape_string = str(shape[0])
    for dim in shape[1:]:
        shape_string = shape_string + '_' + str(dim)
    return shape_string


def get_dump_numpy_dir(graph):
    # mkdir files-dumping directory
    output_dir = graph.graph['cmd_params'].output_dir
    output_model_name = graph.graph['cmd_params'].model_name
    output_model_name = valid_dir_or_file_name(output_model_name)
    dump_numpy_dir = os.path.join(output_dir, output_model_name + '_params')
    if not os.path.exists(dump_numpy_dir):
        os.mkdir(dump_numpy_dir)

    return dump_numpy_dir


# [Eason] dump numpy files for Const nodes
def dump_numpy_files(graph):
    dump_numpy_dir = get_dump_numpy_dir(graph)

    for const_node in graph.get_op_nodes(op='Const'):
        npy_file_name = npy_file_naming(const_node)
        np.save(os.path.join(dump_numpy_dir, npy_file_name), const_node.value)

    return dump_numpy_dir


np_dtype_string_map = {
    np.float16: 'float16',
    np.float32: 'float32',
    np.float64: 'float64',
    np.int8: 'int8',
    np.int32: 'int32',
    np.int64: 'int64',
    np.uint8: 'uint8',
    np.uint64: 'uint64',
}

def np_data_type_to_sting(np_data_type):
    for np_t in np_dtype_string_map.keys():
        if np_t == np_data_type:
            return np_dtype_string_map[np_t]


def store_quan_scale_tensor(des_node: Node, in_port: int, scale: np.ndarray):  
    '''
    Arguments:
        des_node: node that takes the qunatized weight as input
        in_port: the port that takes in this input
        scale: per-channel scale tensor
    '''
    dump_numpy_dir = get_dump_numpy_dir(des_node.graph)
    scale_file_name = 'layer_' + str(des_node.id) + '_input_' + str(in_port) + '_scale'
    np.save(os.path.join(dump_numpy_dir, scale_file_name), scale)

    return scale_file_name

