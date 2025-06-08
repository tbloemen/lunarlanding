from __future__ import annotations
import numpy as np
import torch.nn as nn
import torch
from genepro.node_impl import *


class Multitree(nn.Module):
    def __init__(self, n_trees: int):
        super(Multitree, self).__init__()
        self.n_trees = n_trees
        self.children = []

    def get_output_pt(self, x):
        output = []
        for child in self.children:
            output.append(child.get_output_pt(x).view(-1, 1))

        return torch.cat(output, dim=1)

    def get_subtrees_consts(self):
        constants = []
        for child in self.children:
            constants.extend(
                [
                    node.pt_value
                    for node in child.get_subtree()
                    if isinstance(node, Constant)
                ]
            )
        return constants

    def get_subtrees_internals(self) -> list[Node]:
        internals = []
        for child in self.children:
            internals.extend(
                [
                    node
                    for node in child.get_subtree()
                    if (
                        not isinstance(node, Constant) and not isinstance(node, Feature)
                    )
                ]
            )
        return internals

    def __len__(self) -> int:
        """
        Returns the max length of the trees in the multi-tree
        """
        lens = [len(child) for child in self.children]
        return np.max(lens)

    def get_readable_repr(self) -> str:
        return [child.get_readable_repr() for child in self.children]
