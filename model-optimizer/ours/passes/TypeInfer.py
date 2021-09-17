"""
Make type inference a pass run after pipeline finishes instead of inferring type in emitter 
to avoid error when emitting ops' IRs.
Place this pass after pipeline for the following because
in openvino code, Type Inference is done in ir emitting part, so we assume type inference may need to be done after graph is fixed.
By making it a pass after pipeline finished, we can make type inference run on a fixed graph.
"""

from mo.graph.graph import Graph, Node
from mo.back.replacement import BackReplacementPattern
from ours.passes.ConvActFusion import ConvActFusion
from ours.passes.pass_separator import PipelineFinish

import networkx as nx

import logging as log
from mo.utils.error import Error

# [Eason] make Type Inference as a pass instead of inferring type during ir emition to avoid running into errors when we are emitting ir for op graphs.
# Place this pass after pipeline for the following reason:
# In openvino code, Type Inference is done in ir emitting part, so we assume type inference may need to be done after graph is fixed.
# By making it a pass after pipeline finished, we can make type inference run on a fixed graph.


def type_infer(graph: Graph):
    nodes = list(nx.topological_sort(graph))
    for n in nodes:
        node = Node(graph, n)
        if node.kind == 'op':
            node_name = node.soft_get('name')
            node_type_infer(node)
            log.debug('Type infer for node {}: {}'.format(node_name,
                                                          [port.get_data_type() for port in node.out_ports().values()]))
            """
            Save the precision of input ports in the nodes. It is not possible to get the precision after the port
            re-numbering because the port precision is defined for output port only and for input port it is determined
            with the output port producing data to the input port. When output port id is changed it is not possible to
            determine input port precision.
            """
            for out_port in node.out_ports().values():
                for dest_port in out_port.get_destinations():
                    if not dest_port.node.has_valid('_in_port_precision'):
                        dest_port.node['_in_port_precision'] = {}
                    dest_port.node['_in_port_precision'][dest_port.idx] = out_port.get_data_type()


def node_type_infer(node):
    if node.has_valid('type_infer'):
        node.type_infer(node)
    elif node.has_valid('data_type'):
        node.out_port(0).set_data_type(node.data_type)
    else:
        copy_type_infer(node)


def copy_type_infer(node):
    for out_port in node.out_ports().values():
        connected_in_ports = [port for port in node.in_ports().values() if not port.disconnected()]
        if len(connected_in_ports) != 0:
            data_type = connected_in_ports[0].get_data_type()
            if data_type is not None:
                out_port.set_data_type(data_type)
            else:
                src_node = connected_in_ports[0].get_connection().get_source().node
                node_type_infer(src_node)
                out_port.set_data_type(connected_in_ports[0].get_data_type())
        else:
            raise Error('No input ports of node {} to determine data type'.format(node.soft_get('name')))

class TypeInfer(BackReplacementPattern):
    enabled = True

    def run_after(self):
        return [PipelineFinish]

    def run_before(self):
        return [ConvActFusion]

    def find_and_replace_pattern(self, graph: Graph):
        type_infer(graph)