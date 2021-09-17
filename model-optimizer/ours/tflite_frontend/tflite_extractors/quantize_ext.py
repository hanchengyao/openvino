from ours.our_opset.quantize import Quantize
from mo.front.extractor import FrontExtractorOp

class QuantizeFrontExtractor(FrontExtractorOp):
    op = 'Quantize'
    enabled = True

    @classmethod
    def extract(cls, node):
        # update the attributes of the node
        Quantize.update_node_stat(node, {'output_dtype': node.output_dtype})
        return cls.enabled