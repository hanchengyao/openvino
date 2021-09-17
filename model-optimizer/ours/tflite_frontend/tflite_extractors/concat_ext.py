from mo.front.extractor import FrontExtractorOp
from mo.ops.concat import Concat
    
class ConcatFrontExtractor(FrontExtractorOp):
    op = 'Concat'
    enabled = True

    @classmethod
    def extract(cls, node):
        Concat.update_node_stat(node, {'axis': node.axis})
        return cls.enabled