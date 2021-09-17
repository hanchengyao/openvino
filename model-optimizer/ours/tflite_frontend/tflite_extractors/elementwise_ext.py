import numpy as np

from extensions.ops.elementwise import Add, Sub, Mul, Div, Pow, Less, Equal, Greater, LogicalAnd, LogicalOr, LogicalXor, \
    Round, GreaterEqual, LessEqual
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr
from mo.graph.graph import Node
from mo.ops.eltwise_n import EltwiseNAdd, EltwiseNMax, EltwiseNMin
from mo.ops.power import AttributedPower

class AddFrontExtractor(FrontExtractorOp):
    op = 'Add'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        Add.update_node_stat(node)
        return cls.enabled