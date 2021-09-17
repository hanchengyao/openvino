"""
There are 2 passes, RemoveConstOps and CreateConstNodesReplacement, called before emitting ir.
They remove const nodes in the graph and re-insert new ones corresponding to them, but with new attrs.
Therefore, the attrs of the old const nodes, which the new const nodes don't care about, would not 
be preserved. 
Hence, we don't extract qunatization info directly in the const node.
Instead, we add the qunatization info to the op node that takes this const node as input.
Then, we call this pass after the remove/re-insert const node passes to add quantization info stored 
in the op node to this const node.
This pass also has effect in the situation that the quantized weight is not represented by a const node
(by an op node for instance, if the weight is pre-processed by some op).
"""

from mo.graph.graph import Graph, Node
from mo.back.replacement import BackReplacementPattern

from ours.passes.pass_separator import PipelineFinish

class QuantizationInfo(BackReplacementPattern):
    enabled = False

    def run_after(self):
        return [PipelineFinish]

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        not_operator_op_types = {'Const', 'Parameter', 'Result'}

        for node in graph.nodes:
            node = Node(graph, node)
            if node.kind == 'op' and node.type not in not_operator_op_types:
                if node.has('weight_scale') and node.has('weight_zero_point'):
                    weight_node = node.in_port(1).get_source().node
                    weight_node['weight_scale'] = node.weight_scale
                    weight_node['weight_zero_point'] = node.weight_zero_point
                    node.__delitem__('weight_scale')
                    node.__delitem__('weight_zero_point')
