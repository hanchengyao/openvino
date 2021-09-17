from extensions.ops.activation_ops import *
from mo.front.extractor import FrontExtractorOp

from ours.our_opset.activation_ops_v2 import *

class ReLUExtractor(FrontExtractorOp):
    op = 'ReLU'
    enabled = True

    @classmethod
    def extract(cls, node):
        ReLU.update_node_stat(node)
        return cls.enabled

class ReLU6Extractor(FrontExtractorOp):
    op = 'ReLU6'
    enabled = True

    @classmethod
    def extract(cls, node):
        ReLU6.update_node_stat(node)
        return cls.enabled


class SigmoidExtractor(FrontExtractorOp):
    op = 'Sigmoid'
    enabled = True

    @classmethod
    def extract(cls, node):
        Sigmoid.update_node_stat(node)
        return cls.enabled


class LogisticExtractor(FrontExtractorOp):
    op = 'Logistic'
    enabled = True

    @classmethod
    def extract(cls, node):
        Logistic.update_node_stat(node)
        return cls.enabled