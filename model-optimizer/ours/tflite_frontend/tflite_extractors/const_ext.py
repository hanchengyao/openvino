from mo.front.extractor import FrontExtractorOp
from mo.ops.const import Const


class ConstExtractor(FrontExtractorOp):
    op = 'Const'
    enabled = True

    @classmethod
    def extract(cls, node):
        value = node.value
        attrs = {
            'data_type': value.dtype,
            'value': value
        }
        Const.update_node_stat(node, attrs)
        return cls.enabled