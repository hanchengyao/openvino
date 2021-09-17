from mo.front.onnx.extractors.utils import get_backend_pad
from mo.graph.graph import Node, Graph

from mo.ops.convolution import Convolution


class Convolution_v2(Convolution):
    op = 'Convolution'

    def __init__(self, graph: Graph, attrs: dict):
        attrs['version'] = 'our_opset'  # 'version' of the node will be 'our_opset' since attrs dict in the 
                                        # 3rd arg position are used to update the node's attrs 
                                        # later than the attrs dict in the 2nd arg position.
                                        # see Convolution and Op class initilization method
        super().__init__(graph, attrs)

    def backend_attrs(self):
        def pad_attribute_helper(node: Node, pad_type: str='begin'):
            assert pad_type in ['begin', 'end']
            if not node.has_valid('pad'):
                return None
            pad = get_backend_pad(node.pad, node.spatial_dims, 0 if pad_type == 'begin' else 1)
            if node.has_valid('auto_pad') and node.auto_pad != 'explicit':
                pad = [0 for _ in pad]
            return ','.join(map(str, pad))


        return [
            ('auto_pad', lambda node: node.auto_pad if node.has_valid('auto_pad') else 'explicit'),
            ('strides', lambda node: ','.join(map(str, node['stride'][node.spatial_dims]))),
            ('dilations', lambda node: ','.join(map(str, node['dilation'][node.spatial_dims]))),
            ('pads_begin', lambda node: pad_attribute_helper(node, 'begin')),
            ('pads_end', lambda node: pad_attribute_helper(node, 'end')),

            # for Backpropdata operations only - according to spec
            ('output_padding', lambda node: ','.join(map(str, node.output_padding[node.spatial_dims])) \
                if node.has_valid('output_padding') and node.type in
                    ('GroupConvolutionBackpropData', 'ConvolutionBackpropData') else None),

            # for BinaryConvolution only
            'pad_value',
            'mode',

            # [Eason] add 'bias_add' and 'act_func' to the attrs of conv's xml layer
            ('bias_add', lambda node: str(node.bias_term)),
            'act_func',
            # conv is mac-accountable
            'macs',
        ]