# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import networkx as nx


class ClassicalLinkPredictionModel:
    def __init__(self, model_name):
        self.model_name = model_name

    def __repr__(self):
        return f"ClassicalLinkPredictionModel({self.model_name})"

    def run(self,
            G: nx.Graph,
            ebunch):
        assert isinstance(G, nx.Graph), type(G)
        assert not G.is_directed()

        if self.model_name == "adamic_adar":
            return nx.adamic_adar_index(G, ebunch)
        elif self.model_name == "jaccard":
            return nx.jaccard_coefficient(G, ebunch)
        elif self.model_name == "resource_allocation":
            return nx.resource_allocation_index(G, ebunch)
        else:
            raise ValueError(f"Unavailable model: {self.model_name}")

    def __call__(self,
                 G: nx.Graph,
                 ebunch):
        return self.run(G, ebunch)
