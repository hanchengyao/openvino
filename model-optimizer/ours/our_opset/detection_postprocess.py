import numpy as np

from mo.front.common.partial_infer.multi_box_detection import multi_box_detection_infer
from mo.front.extractor import bool_to_str
from mo.graph.graph import Graph, Node
from mo.ops.op import Op


class DetectionPostprocess(Op):
    op = 'DetectionPostprocess'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': self.op,
            'op': self.op,
            'version': 'our_opset',
            'in_ports_count': 3,
            'out_ports_count': 4,
            'infer': self.infer,
            # 'input_width': 1,
            # 'input_height': 1,
            # 'normalized': True,
            # 'share_location': True,
            # 'clip_after_nms': False,
            # 'clip_before_nms': False,
            # 'decrease_label_id': False,
            # 'variance_encoded_in_target': False,
            # 'type_infer': self.type_infer,
        }, attrs)

    def supported_attrs(self):
        return [
            'max_detections',
            'max_classes_per_detection',
            'nms_score_threshold',
            'nms_iou_threshold',
            'num_classes',
            'x_scale',
            'y_scale',
            'h_scale',
            'w_scale',
            'use_regular_nms',
            # 'background_label_id',
            # ('clip_after_nms', lambda node: bool_to_str(node, 'clip_after_nms')),
            # ('clip_before_nms', lambda node: bool_to_str(node, 'clip_before_nms')),
            # 'code_type',
            # 'confidence_threshold',
            # ('decrease_label_id', lambda node: bool_to_str(node, 'decrease_label_id')),
            # 'input_height',
            # 'input_width',
            # 'keep_top_k',
            # 'nms_threshold',
            # ('normalized', lambda node: bool_to_str(node, 'normalized')),
            # 'num_classes',
            # ('share_location', lambda node: bool_to_str(node, 'share_location')),
            # 'top_k',
            # ('variance_encoded_in_target', lambda node: bool_to_str(node, 'variance_encoded_in_target')),
            # 'objectness_score',
        ]

    # @staticmethod
    # def type_infer(node: Node):
    #     node.out_port(0).set_data_type(np.float32)
    #     node.out_port(1).set_data_type(np.float32)
    #     node.out_port(2).set_data_type(np.float32)
    #     node.out_port(3).set_data_type(np.float32)

    @staticmethod
    def infer(node: Node):
        """
        TFLite_Detection_PostProcess custom op node has four outputs:
        detection_boxes: a float32 tensor of shape [1, num_boxes, 4] with box
        locations
        detection_classes: a float32 tensor of shape [1, num_boxes]
        with class indices
        detection_scores: a float32 tensor of shape [1, num_boxes]
        with class scores
        num_boxes: a float32 tensor of size 1 containing the number of detected boxes
        """

        loc_shape = node.in_node(0).shape
        conf_shape = node.in_node(1).shape
        prior_boxes_shape = node.in_node(2).shape
        num_boxes = prior_boxes_shape[0]
        node.out_node(0).shape = np.array([1, num_boxes, 4], dtype=np.int64)
        node.out_node(1).shape = np.array([1, num_boxes], dtype=np.int64)
        node.out_node(2).shape = np.array([1, num_boxes], dtype=np.int64)
        node.out_node(3).shape = np.array([num_boxes], dtype=np.int64)
