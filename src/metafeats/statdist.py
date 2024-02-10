# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math

import numpy as np
from scipy import stats


def meta_features_per_structural_property(x):
    """
    Computes meta-features over a vector of numbers. In the case of graphs, this is typically a vector of values for each node, or a distribution of values, etc.
    For instance, degrees, k-core numbers, eccentricity values, wedges per node, triangles per node, or triangles per edge, etc.

    INPUT: Per-node/edge structural feature where x[i]=value of structural feature for node/edge i (e.g., degree of node i, triangle count of node i, pagerank of node i)
    OUTPUT: k-dimensional vector representation for the per-node/edge structural feature
    """
    mean_x = np.nanmean(x)
    median_x = np.nanmedian(x)
    min_x = np.nanmin(x)
    max_x = np.nanmax(x)
    var_x = np.nanvar(x)
    std_x = np.nanstd(x)
    skew_x = stats.skew(x)
    # hyperskewness = stats.moment(x, moment=5)

    kur_fisher_x = stats.kurtosis(x, fisher=True)
    kur_pearson_x = stats.kurtosis(x, fisher=False)
    kur_fisher_bias_x = stats.kurtosis(x, fisher=True, bias=False)
    kur_pearson_bias_x = stats.kurtosis(x, fisher=False, bias=False)

    if False:
        """
        As sample size increases, n-th moment and n-th k-statistic converge to the same number
        (although they aren’t identical).
        In the case of the normal distribution, they converge to zero.
        """
        kstat_one_x = stats.kstat(x, 1)
        kstat_two_x = stats.kstat(x, 2)
        kstat_three_x = stats.kstat(x, 3)
        kstat_four_x = stats.kstat(x, 4)

    geometric_mean = stats.gmean(x)
    harmonic_mean = stats.hmean(x + 0.0000000001)

    coeff_of_variation = stats.variation(x)
    mad = np.nanmedian(np.abs(x - np.nanmedian(x)))
    avg_abs_dev = np.nanmean(np.abs(x - np.nanmean(x)))

    ent = stats.entropy(x)

    """
    $\frac{H(\vx)}{\log_2 n}$
    """
    norm_entropy = ent / (np.log2(x.shape[0]) * 1.)  # $\frac{H(\vx)}{\log_2 n}$

    gini_coeff = gini(x)

    if False:
        m5 = stats.moment(x, moment=5)
        m6 = stats.moment(x, moment=6)
        m7 = stats.moment(x, moment=7)
        m8 = stats.moment(x, moment=8)
        m9 = stats.moment(x, moment=9)
        m10 = stats.moment(x, moment=10)

    # iqr = stats.iqr(x)
    idx_sorted = np.argsort(x, axis=0)
    x_vec_sorted = x[idx_sorted]

    n_tmp = x_vec_sorted.shape[0]
    med_idx = int(math.floor(n_tmp / 2))
    quartiles_idx = int(math.floor(n_tmp / 4))
    Q1_idx = med_idx - quartiles_idx
    Q3_idx = med_idx + quartiles_idx

    # median = np.median(x_vec_sorted)
    q1 = x_vec_sorted[Q1_idx]
    q3 = x_vec_sorted[Q3_idx]
    iqr = q3 - q1

    alpha_factor = 1.5
    lb = q1 - (alpha_factor * iqr)
    ub = q3 + (alpha_factor * iqr)

    num_ub_outliers = 0
    num_lb_outliers = 0
    num_outliers = 0
    for x_val in x:
        if x_val > ub:
            num_ub_outliers += 1
        elif x_val < lb:
            num_lb_outliers += 1
        if x_val < lb or x_val > ub: num_outliers += 1

    perc_ub_outliers = num_ub_outliers / (n_tmp * 1.)
    perc_lb_outliers = num_lb_outliers / (n_tmp * 1.)
    perc_outliers = num_outliers / (n_tmp * 1.)

    alpha_factor = 3.0
    lb_three = q1 - (alpha_factor * iqr)
    ub_three = q3 + (alpha_factor * iqr)

    num_ub_outliers_three = 0
    num_lb_outliers_three = 0
    num_outliers_three = 0
    for x_val in x:
        if x_val > ub_three:
            num_ub_outliers_three += 1
        elif x_val < lb_three:
            num_lb_outliers_three += 1
        if x_val < lb_three or x_val > ub_three: num_outliers_three += 1

    perc_ub_outliers_three = num_ub_outliers_three / (n_tmp * 1.)
    perc_lb_outliers_three = num_lb_outliers_three / (n_tmp * 1.)
    perc_outliers_three = num_outliers_three / (n_tmp * 1.)

    std_factor_vec = [1.0, 2.0, 3.0]
    outlier_std_metafeatures = []
    for fac in std_factor_vec:
        lb_std = mean_x - fac * std_x
        ub_std = mean_x + fac * std_x
        num_ub_outliers_std = 0
        num_lb_outliers_std = 0
        num_outliers_std = 0
        for x_val in x:
            if x_val > ub_std:
                num_ub_outliers_std += 1
            elif x_val < lb_std:
                num_lb_outliers_std += 1
            if x_val < lb_std or x_val > ub_std: num_outliers_std += 1

        perc_ub_outliers_std = num_ub_outliers_std / (n_tmp * 1.)
        perc_lb_outliers_std = num_lb_outliers_std / (n_tmp * 1.)
        perc_outliers_std = num_outliers_std / (n_tmp * 1.)

        outlier_std_metafeatures.extend([num_outliers_std, num_lb_outliers_std, num_ub_outliers_std])
        outlier_std_metafeatures.extend([perc_outliers_std, perc_lb_outliers_std, perc_ub_outliers_std])

    mode, mode_count = stats.mode(x)
    mode = mode[0]
    mode_count = mode_count[0]

    mode_frac = mode_count / (x.shape[0] * 1.)

    # spearman_rho, spearman_pval = stats.spearmanr(x, x_vec_sorted)
    # pearson_r, pearson_pval = stats.pearsonr(x, x_vec_sorted)
    # tau, p_value = stats.kendalltau(x, x_vec_sorted)

    """
    Compute sequence corr of counts per bin
    """
    quartile_dispersion_coeff = (q3 - q1) / (q3 + q1) * 1.
    signal_to_noise_ratio = (mean_x * mean_x) / (std_x * std_x * 1.)
    efficiency_ratio = (std_x * std_x) / (mean_x * mean_x * 1.)
    var_to_mean_ratio = (std_x * std_x) / (mean_x * 1.)

    var_features = [mean_x, median_x, min_x, max_x, var_x, std_x]
    var_features.extend([ent, norm_entropy, gini_coeff, coeff_of_variation, mad, avg_abs_dev])
    var_features.extend([quartile_dispersion_coeff, signal_to_noise_ratio, efficiency_ratio, var_to_mean_ratio])

    var_features.extend([skew_x, kur_fisher_x, kur_pearson_x, kur_fisher_bias_x, kur_pearson_bias_x])

    var_features.extend([geometric_mean, harmonic_mean])
    var_features.extend([perc_outliers, perc_lb_outliers, perc_ub_outliers])
    var_features.extend([num_outliers, num_lb_outliers, num_ub_outliers])
    var_features.extend([perc_outliers_three, perc_lb_outliers_three, perc_ub_outliers_three])
    var_features.extend([num_outliers_three, num_lb_outliers_three, num_ub_outliers_three])
    var_features.extend([lb_three, ub_three])
    var_features.extend([iqr, q1, q3, lb, ub])
    var_features.extend(outlier_std_metafeatures)
    var_features.extend([mode, mode_count, mode_frac])

    structural_prop_meta_features = np.array(var_features).flatten()

    # if any feature values are -inf/inf, then set them to 0
    idx_inf = np.argwhere(np.isinf(structural_prop_meta_features)).flatten()
    structural_prop_meta_features[idx_inf] = 0

    idx_nan = np.argwhere(np.isnan(structural_prop_meta_features)).flatten()
    structural_prop_meta_features[idx_nan] = 0

    return structural_prop_meta_features


def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    array = array.copy()
    array = np.array(array, dtype=float)
    array = array.flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)
    array += 0.0000001
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))



if __name__ == '__main__':
    x = np.random.rand(30)
    m = meta_features_per_structural_property(x)
    print(m, m.shape)
