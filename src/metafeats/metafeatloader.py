# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from enum import Enum, unique
from timeit import default_timer as timer

import numpy as np

import settings
from graphs.graphset import Graph
from utils import logger


@unique
class MetaFeatType(Enum):
    COMPACT = "compact"                         # num feats: 58
    REGULAR = "regular"                         # num feats: 318
    GRAPHLETS_COMPLEX = "graphlets_complex"     # num feats: 756
    TINY = "tiny"                               # num feats: 13
    REGULAR_GRAPHLETS = "regular_graphlets"     # num feats: 1074 (318+756)
    ALL = "all"                                 # num feats: 1132 (58+318+756)


class MetaGraphFeatLoader:
    glet_input_graph_fn = "glet_input_graph.csv"
    glet_output_orbits_fn = "glet_output_orbits.csv"

    def __init__(self,
                 graph: Graph,
                 meta_feat_type: MetaFeatType):
        self.graph = graph
        self.meta_feat_type = meta_feat_type

    @property
    def meta_graph_feat_root(self):
        return settings.META_FEAT_ROOT / f"{self.graph.name}"

    def meta_graph_feat_path(self, meta_feat_type=None):
        if meta_feat_type is None:
            meta_feat_type = self.meta_feat_type
        return self.meta_graph_feat_root / f"m_{meta_feat_type.value}.npy"

    def meta_graph_feat_runtime_path(self, meta_feat_type=None):
        if meta_feat_type is None:
            meta_feat_type = self.meta_feat_type
        return self.meta_graph_feat_root / f"m_{meta_feat_type.value}-runtime.txt"

    def load(self, meta_feat_type=None):
        if meta_feat_type is None:
            meta_feat_type = self.meta_feat_type

        if meta_feat_type is MetaFeatType.REGULAR_GRAPHLETS:
            return np.concatenate([self.load(MetaFeatType.REGULAR),
                                   self.load(MetaFeatType.GRAPHLETS_COMPLEX)])
        elif meta_feat_type is MetaFeatType.ALL:
            return np.concatenate([self.load(MetaFeatType.COMPACT),
                                   self.load(MetaFeatType.REGULAR),
                                   self.load(MetaFeatType.GRAPHLETS_COMPLEX)])

        if not self.meta_graph_feat_path(meta_feat_type).exists():
            self.generate()

        assert self.meta_graph_feat_path(meta_feat_type).is_file()
        return np.load(self.meta_graph_feat_path(meta_feat_type))

    def generate(self, meta_feat_type=None):
        if meta_feat_type is None:
            meta_feat_type = self.meta_feat_type
        self.meta_graph_feat_path(meta_feat_type).parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating {self.meta_feat_type.value} meta feats for graph {self.graph.name}...")
        start = timer()

        data = self.graph.pyg_graph()
        if meta_feat_type is MetaFeatType.COMPACT:
            from metafeats.compact_meta_feats import compact_meta_graph_features
            meta_feats = compact_meta_graph_features(
                data=data,
                glet_input_graph_fn=self.meta_graph_feat_root / self.glet_input_graph_fn,
                output_orbits_fn=self.meta_graph_feat_root / self.glet_output_orbits_fn,
            )[0]
        elif meta_feat_type is MetaFeatType.REGULAR:
            from metafeats.regular_meta_feats import regular_meta_graph_features
            meta_feats = regular_meta_graph_features(data=data)
        elif meta_feat_type is MetaFeatType.TINY:
            from metafeats.tiny_meta_feats import tiny_meta_graph_features
            meta_feats = tiny_meta_graph_features(data=data)
        elif meta_feat_type is MetaFeatType.GRAPHLETS_COMPLEX:
            from metafeats.graphlets_meta_feats import graphlets_meta_graph_features
            meta_feats = graphlets_meta_graph_features(
                data=data,
                glet_input_graph_fn=self.meta_graph_feat_root / self.glet_input_graph_fn,
                output_orbits_fn=self.meta_graph_feat_root / self.glet_output_orbits_fn,
            )
        else:
            raise ValueError(f"Invalid type: {meta_feat_type}")

        logger.info(f"Generated {meta_feat_type.value} meta feats for graph {self.graph.name}.")
        elapsed_secs = timer() - start
        runtime_path = self.meta_graph_feat_runtime_path(meta_feat_type)
        with runtime_path.open('w') as f:
            f.write(f"{elapsed_secs:.6f}")
            logger.info(f"Elapsed time for {meta_feat_type.value} meta-feature saved to {runtime_path}")

        feat_path = self.meta_graph_feat_path(meta_feat_type)
        np.save(file=feat_path.open('wb'), arr=meta_feats)
        logger.info(f"{meta_feat_type.value} meta-feature (num feats={len(meta_feats)}) "
                    f"saved to {feat_path}")

        return self


def generate_feats_all_graphs():
    from graphs.graphset import GraphSet
    graphset = GraphSet(data_sources=['netrepo', 'pyg'], sort_graphs='num_edges')
    graphs = graphset.graphs
    print("# graphs:", len(graphs))

    for i, graph in enumerate(graphs):
        logger.info(f"Processing {graph.name} ({i + 1}/{len(graphs)})...")
        start = timer()
        MetaGraphFeatLoader(graph, MetaFeatType.TINY).load()
        MetaGraphFeatLoader(graph, MetaFeatType.COMPACT).load()
        MetaGraphFeatLoader(graph, MetaFeatType.REGULAR).load()
        MetaGraphFeatLoader(graph, MetaFeatType.GRAPHLETS_COMPLEX).load()
        logger.info(f"Processed {graph.name} ({i + 1}/{len(graphs)}): {timer() - start} secs.")


if __name__ == '__main__':
    # from graphs.netrepo_graphs.netrepo_unlabeled_graphs import load_graphs
    #
    # graphs = load_graphs()
    # graph = graphs[20]  # bio-CE-GT
    #
    # simple_feats = MetaGraphFeatLoader(graph, MetaFeatType.COMPACT).load()
    # print("simple_feats:", simple_feats.shape)

    # regular_feats = MetaGraphFeatLoader(graph, MetaFeatType.REGULAR).load()
    # print("regular_feats:", regular_feats.shape)
    #
    # graphlets_complex_feats = MetaGraphFeatLoader(graph, MetaFeatType.GRAPHLETS_COMPLEX).load()
    # print("graphlets_complex_feats:", graphlets_complex_feats.shape)
    #
    # all_feats = MetaGraphFeatLoader(graph, MetaFeatType.ALL).load()
    # print("all_feats:", all_feats.shape)
    # assert len(all_feats) == sum(map(lambda x: len(x), [simple_feats, regular_feats, graphlets_complex_feats]))

    generate_feats_all_graphs()
