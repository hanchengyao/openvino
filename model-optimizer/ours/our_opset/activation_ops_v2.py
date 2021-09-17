import numpy as np
from mo.graph.graph import Graph, Node
from extensions.ops.activation_ops import Activation


class Activation_v2(Activation):
    version = 'our_opset'

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, attrs)


class Logistic(Activation_v2):
    op = 'Logistic'
    operation = staticmethod(lambda x: 1 / (1 + np.exp(-x)))