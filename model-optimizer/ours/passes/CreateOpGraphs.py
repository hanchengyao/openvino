"""
This pass generates a graph for each op node (except for 'Const', 'Parameter', 'Result' nodes) in the model
graph.
The graph generated for each op retains every in/output node to/from this op and the edges connecting them
in the model graph. 
Therefore, each op graph can be seen as a model which has only one operation(the op) in it.  
And since this pass is run after the original pipeline finished, the op graphs are well-prepared to be
passed to emitter(either 1 or 2) to generate IRs for these op graphs. 
"""

from argparse import Namespace
from mo.graph.graph import Graph, Node
from mo.back.replacement import BackReplacementPattern
from ours.passes.ConvActFusion import ConvActFusion

import os
from ours.utils import valid_dir_or_file_name

class CreateOpGraphs(BackReplacementPattern):
    '''
    This pass generates a graph for each op node (except for 'Const', 'Parameter', 'Result' nodes) in the model
    graph.
    The graph generated for each op retains every in/output node to/from this op and the edges connecting them
    in the model graph. 
    Therefore, each op graph can be seen as a model which has only one operation(the op) in it.  
    And since this pass is run after the original pipeline finished, the op graphs are well-prepared to be
    passed to emitter(either 1 or 2) to generate IRs for these op graphs. 
    '''
    enabled = True

    def run_after(self):
        return [ConvActFusion]

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        # create a dir to store all op irs. the dir follows the dir structure and naming:
        # model_graph_output_dir/model_name_op_ir/
        model_graph_output_dir = graph.graph['cmd_params'].output_dir
        store_op_ir_dir = os.path.join(model_graph_output_dir, valid_dir_or_file_name(graph.name) + '_op_ir')
        if not os.path.exists(store_op_ir_dir):
            os.mkdir(store_op_ir_dir)

        op_graphs_list = []  # to store all op graphs for all op in the model graph
        not_operator_op_types = {'Const', 'Parameter', 'Result'}
        for node in graph.nodes:
            op_graph = graph.copy()  # make a copy of the graph to create op_graph
            node_name = node
            node = Node(op_graph, node)
            # build a graph for each op in the model graph
            # we construct a op_graph that contain its input, itself, and its output, and retain the data flow between them.
            if node.kind == 'op' and node.type not in not_operator_op_types:
                op_graph_nodes_list = [node_name]

                # gather this node's input nodes and the data nodes in between: 
                # (input_node ---> data_node) ---> this_op_node. 
                in_data_edges = list(node.graph.in_edges(node.id, data=False))
                for in_data_edge in in_data_edges:
                    in_data_name, _ = in_data_edge
                    assert(node.graph.node[in_data_name]['kind'] == 'data')
                    op_graph_nodes_list.append(in_data_name)
                    in_node_edges = list(node.graph.in_edges(in_data_name, data=False))
                    assert (len(in_node_edges) == 1)
                    in_node_name, _ = in_node_edges[0]
                    assert(node.graph.node[in_node_name]['kind'] == 'op')
                    op_graph_nodes_list.append(in_node_name)

                # gather this node's output nodes and the data nodes in between: 
                # this_op_node ---> (data_node ---> output_node)
                out_data_edges = list(node.graph.out_edges(node.id, data=False))
                for out_data_edge in out_data_edges:
                    _, out_data_name = out_data_edge
                    assert(node.graph.node[out_data_name]['kind'] == 'data')
                    op_graph_nodes_list.append(out_data_name)
                    out_node_edges = list(node.graph.out_edges(out_data_name, data=False))
                    for out_node_edge in out_node_edges:
                        _, out_node_name = out_node_edge
                        assert(node.graph.node[out_node_name]['kind'] == 'op')
                        op_graph_nodes_list.append(out_node_name)

                # remove nodes unrelated to this op node
                nodes_to_be_removed = [x for x in list(op_graph.nodes) if x not in op_graph_nodes_list]
                op_graph.remove_nodes_from(nodes_to_be_removed)
                op_graph.name = node.name
                op_ir_dir = valid_dir_or_file_name(node.name) + '_ir'
                op_graph.graph['cmd_params'] = Namespace(**vars(graph.graph['cmd_params']))
                op_graph.graph['cmd_params'].output_dir = os.path.join(store_op_ir_dir, op_ir_dir)
                if not os.path.exists(op_graph.graph['cmd_params'].output_dir):
                    os.mkdir(op_graph.graph['cmd_params'].output_dir)
                op_graph.graph['cmd_params'].model_name = valid_dir_or_file_name(node.name)
                op_graphs_list.append(op_graph)

        graph.graph['op_graphs_list'] = op_graphs_list