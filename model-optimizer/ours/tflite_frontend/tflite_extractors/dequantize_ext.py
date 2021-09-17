from ours.our_opset.dequantize import Dequantize
from mo.front.extractor import FrontExtractorOp

class DequantizeFrontExtractor(FrontExtractorOp):
    op = 'Dequantize'
    enabled = True

    @classmethod
    def extract(cls, node):
        # update the attributes of the node
        Dequantize.update_node_stat(node, {'output_dtype': node.output_dtype})
        return cls.enabled