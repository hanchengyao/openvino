import numpy as np

from mo.graph.graph import Graph, Node
from mo.ops.op import Op
from mo.front.common.partial_infer.elemental import copy_shape_infer


class Quantize(Op):
    op = 'Quantize'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': self.op,
            'op': self.op,
            'version': 'our_opset',
            'in_ports_count': 1,
            'out_ports_count': 1,
            'infer': copy_shape_infer,
            'type_infer': self.type_infer,
        }, attrs)

    def supported_attrs(self):
        return [
            'scale',
            'zero_point'
        ]

    @staticmethod
    def type_infer(node: Node):
        node.out_port(0).set_data_type(node.output_dtype)

    @staticmethod
    def infer(node: Node):
        # node.out_node(0).shape = node.in_node(0).shape
        node.out_port(0).data.set_shape(node.in_port(0).data.get_shape())
