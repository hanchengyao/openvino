from ours.our_opset.detection_postprocess import DetectionPostprocess
from mo.front.extractor import FrontExtractorOp

class DetectionPostprocessFrontExtractor(FrontExtractorOp):
    op = 'DetectionPostprocess'
    enabled = True

    @classmethod
    def extract(cls, node):
        # update the attributes of the node
        # DetectionPostprocess is a tflite custom op, so the attrs of this op are tflite-specific and 
        # we use exactly those attrs parsed from tflite_detection_postprocess to update node attrs.
        DetectionPostprocess.update_node_stat(node)
        return cls.enabled