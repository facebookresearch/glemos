# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import json
import pprint

import numpy as np

np.set_printoptions(precision=3, suppress=True)
from utils import setup_cuda, logger, set_seed, output_results

from testbeds import FullyObservedPerfTestbed, PartiallyObservedPerfTestbed, OutOfDomainTestbed, \
    SmallToLargeTestbed, CrossTaskTestbed


def main(args):
    testbed = get_testbed(args)

    if args.testbed == 'cross-task':
        num_models = len(testbed.get_common_models())
        num_meta_feats = testbed.source_meta_feat_mat.shape[1]
        assert testbed.source_meta_feat_mat.shape[1] == testbed.target_meta_feat_mat.shape[1]
    else:
        num_graphs, num_models = testbed.perf_mat.shape
        num_meta_feats = testbed.meta_feat_mat.shape[1]

    methods = get_model_selection_methods(args.k_dim, num_models, num_meta_feats)
    logger.info("=" * 120)
    logger.info(f"Model selection methods: {', '.join([m.name for m in methods])}")
    logger.info("=" * 120)

    P_splits, M_splits, D_splits = testbed.load()
    result_dir = testbed.get_result_dir_path()
    result_dir.mkdir(parents=True, exist_ok=True)

    for m in methods:
        logger.info(f"Running {m.name} method...")

        """skip method if it has already been evaluated"""
        eval_dict_path = result_dir / f"eval_dict-{m.name}.json"
        if eval_dict_path.exists():
            logger.info(f"skipping method {m.name} as it has already been evaluated: {eval_dict_path}.")
            with eval_dict_path.open('r') as f:
                m.eval_dict = json.load(f)
                continue

        """reset seeds for each model selection method"""
        set_seed(args.seed)

        """train and test"""
        P_test_list, P_test_hat_list = [], []
        for M_dict, P_dict, D_dict in zip(M_splits, P_splits, D_splits):
            M_train, M_test = M_dict["train"], M_dict["test"]
            P_train, P_train_imputed, P_train_full, P_test = \
                P_dict["train"], P_dict["train_imputed"], P_dict["train_full"], P_dict["test"]
            D_train, D_test = D_dict["train"], D_dict["test"]
            P_test_list.append(P_test)

            P_test_hat = m.fit_predict(M_train, M_test, P_train, P_train_imputed, P_train_full, P_test, D_train)
            P_test_hat_list.append(P_test_hat)

        with eval_dict_path.open('w') as f:
            json.dump(m.eval_dict, f)

        # save P_test and P_test_hat for later analysis
        np.save(result_dir / f"P_test_true-{m.name}.npy", np.concatenate(P_test_list))
        np.save(result_dir / f"P_test_pred-{m.name}.npy", np.concatenate(P_test_hat_list))
        if hasattr(m, "predict_times"):
            # noinspection PyTypeChecker
            np.savetxt(result_dir / f"P_times-{m.name}.csv", np.array(m.predict_times), delimiter=",")

    method_names = [m.name for m in methods]
    method_eval = [m.eval_dict for m in methods]

    output_results(result_dir, testbed, method_names, method_eval, testbed.meta_feat, args)


def get_testbed(args):
    from testbeds.workspace import perf_matrices, meta_features, y_graph_domain, graph_names, all_models

    if args.testbed == 'fully-observed':
        testbed = FullyObservedPerfTestbed(
            task=args.task, perf_metric=args.perf_metric, perf_matrices=perf_matrices,
            meta_feat=args.meta_feat, meta_features=meta_features,
            n_splits=args.n_splits, graph_names=graph_names,
            all_models=all_models, graph_domain=y_graph_domain,
        )
    elif args.testbed == 'partially-observed':
        assert args.perf_sparsity > 0, args.perf_sparsity
        testbed = PartiallyObservedPerfTestbed(
            task=args.task, perf_metric=args.perf_metric, perf_matrices=perf_matrices,
            meta_feat=args.meta_feat, meta_features=meta_features,
            perf_sparsity=args.perf_sparsity, n_splits=args.n_splits,
            graph_names=graph_names, all_models=all_models,
            graph_domain=y_graph_domain
        )
    elif args.testbed == 'out-of-domain-kfold':
        testbed = OutOfDomainTestbed(
            task=args.task, perf_metric=args.perf_metric, perf_matrices=perf_matrices,
            meta_feat=args.meta_feat, meta_features=meta_features,
            n_splits=args.n_splits, graph_domain=y_graph_domain,
            graph_names=graph_names, all_models=all_models,
            test_mode=OutOfDomainTestbed.TEST_MODE_KFOLD,
        )
    elif args.testbed == 'out-of-domain-logo':  # logo: leave-one-group-out
        testbed = OutOfDomainTestbed(
            task=args.task, perf_metric=args.perf_metric, perf_matrices=perf_matrices,
            meta_feat=args.meta_feat, meta_features=meta_features,
            n_splits=None, graph_domain=y_graph_domain,
            graph_names=graph_names, all_models=all_models,
            test_mode=OutOfDomainTestbed.TEST_MODE_LOGO,
        )
    elif args.testbed == 'small-to-large':
        testbed = SmallToLargeTestbed(
            task=args.task, perf_metric=args.perf_metric, perf_matrices=perf_matrices,
            meta_feat=args.meta_feat, meta_features=meta_features,
            n_splits=args.n_splits, graph_names=graph_names, all_models=all_models,
        )
    elif args.testbed == 'cross-task':
        testbed = CrossTaskTestbed(
            source_task=args.source_task,
            source_perf_metric=args.source_perf_metric,
            target_task=args.target_task,
            target_perf_metric=args.target_perf_metric,
            perf_matrices=perf_matrices,
            meta_feat=args.meta_feat,
            meta_features=meta_features,
            graph_names=graph_names,
            models=all_models,
        )
    else:
        raise ValueError(f"Unavailable testbed: {args.testbed}")

    return testbed


# noinspection PyUnusedLocal
def get_model_selection_methods(k_dim, num_models, num_meta_feats):
    from model_selection_methods import RandomSelection, ISAC, ARGOSMART, GlobalBestAveragePerf, GlobalBestAverageRank, \
        ALORS, SupervisedSurrogate, NCF, MetaOD, MetaGL

    rand_selection = RandomSelection(name="RandomSelection")
    gb_perf = GlobalBestAveragePerf(name="GB-AvgPerf")
    gb_rank = GlobalBestAverageRank(name="GB-AvgRank")
    isac = ISAC(name="ISAC")
    argosmart = ARGOSMART(name="AS", num_models=num_models)
    alors = ALORS(name="ALORS", k_dim=k_dim)
    s2 = SupervisedSurrogate(name="S2", hid_dim=k_dim, num_meta_feats=num_meta_feats,
                             num_models=num_models, device=args.device, epochs=args.epochs)
    ncf = NCF(name="NCF", hid_dim=k_dim, model_feats_dim=k_dim * 4,
              num_meta_feats=num_meta_feats, num_models=num_models, device=args.device, epochs=args.epochs)
    metaod = MetaOD(name="MetaOD", n_factors=k_dim)
    metagl = MetaGL(num_models=num_models, metafeats_dim=num_meta_feats, epochs=args.epochs,
                    device=args.device, hid_dim=args.k_dim)

    all_model_selection_methods = [
        rand_selection,
        gb_perf,
        gb_rank,
        isac,
        argosmart,
        alors,
        s2,
        ncf,
        metaod,
        metagl,
    ]

    if not args.model_selection_methods:
        model_selection_methods = all_model_selection_methods
    else:
        model_selection_methods = []
        for m in all_model_selection_methods:
            for method in args.model_selection_methods:
                if method.lower() in m.name.lower():
                    model_selection_methods.append(m)
                    break
    assert len(model_selection_methods) == len(set(model_selection_methods))

    return model_selection_methods


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set to -1 to use CPU.")
    parser.add_argument("--seed", type=int, default=1337,
                        help="random seed")
    parser.add_argument("--task", type=str, default='link-pred',
                        choices=['link-pred', 'node-class'],  # link prediction, node classification
                        help="name of the graph learning task for model selection")
    parser.add_argument("--source-task", type=str,  # for cross-task testbed
                        choices=['link-pred', 'node-class'],  # link prediction, node classification
                        help="name of the source graph learning task for training model selection algorithms (used for cross-task testbed)")
    parser.add_argument("--target-task", type=str,  # for cross-task testbed
                        choices=['link-pred', 'node-class'],  # link prediction, node classification
                        help="name of the target graph learning task for training model selection algorithms (used for cross-task testbed)")
    parser.add_argument("--testbed", type=str, default='fully-observed',
                        choices=['fully-observed', 'partially-observed', 'out-of-domain-kfold', 'out-of-domain-logo',
                                 'small-to-large', 'cross-task'],
                        help="name of the testbed")
    parser.add_argument("--perf-metric", type=str,
                        default='map', choices=['map', 'auc', 'ndcg'],
                        help="performance metric")
    parser.add_argument("--source-perf-metric", type=str,  # for cross-task testbed
                        default='map', choices=['map', 'auc', 'ndcg'],
                        help="performance metric for the source task (used for cross-task testbed)")
    parser.add_argument("--target-perf-metric", type=str,  # for cross-task testbed
                        default='map', choices=['map', 'auc', 'ndcg'],
                        help="performance metric for the target test (used for cross-task testbed)")
    parser.add_argument("--meta-feat", type=str, default='regular',
                        choices=['regular', 'compact', 'tiny', 'graphlets_complex', 'regular_graphlets', 'all'],
                        help="name of the meta-feature set")
    parser.add_argument("--perf-sparsity", type=float, default=0.0,
                        help="sparsity of the performance matrix (i.e., percentage of non-nans in the performance matrix)")
    parser.add_argument("--n-splits", type=int, default=5,
                        help="number of splits for k-fold cross validation (not applicable to some testbeds)")
    parser.add_argument("--model-selection-methods", type=str, default=None,
                        help='comma separated names of model selection method to run (e.g., "isac,s2"')
    parser.add_argument("--epochs", type=int, default=500,
                        help="number of training epochs")
    parser.add_argument("--k-dim", type=int, default=32,
                        help="embedding dimension")

    args = parser.parse_args()
    set_seed(args.seed)
    setup_cuda(args)

    if args.model_selection_methods is not None:
        # noinspection PyUnresolvedReferences
        args.model_selection_methods = [name.strip() for name in args.model_selection_methods.strip().split(",")]
    else:
        args.model_selection_methods = []

    print("\n[Arguments]\n" + pprint.pformat(args.__dict__))
    main(args)
