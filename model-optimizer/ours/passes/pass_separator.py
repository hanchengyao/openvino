"""
Run this pass after all the openvino original passes are done 
This pass acts as a separator of our passes and the original passes
"""

from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph
from extensions.back.blob_normalizer import BlobNormalizer

# run our pass after all the openvino original passes are done
class PipelineFinish(BackReplacementPattern):
    enabled = True
    force_clean_up = True

    def run_after(self):
        return [BlobNormalizer]

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        pass