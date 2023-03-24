# ----- Standard Imports
import itertools
from collections import defaultdict

# ----- Third Party Imports
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold, ShuffleSplit

# ----- Library Imports
from fair_robust_classifiers.cross_validation.strategies import SimpleBestMetricStrategy, BestDDPOnUtilityPercentile


SCALER = MinMaxScaler(feature_range=(0,1))
#SCALER = StandardScaler()

VALIDATION_SPLIT_STRATEGY = ShuffleSplit(n_splits=10, test_size=.2, random_state=42)
# VALIDATION_SPLIT_STRATEGY = KFold(n_splits=5, shuffle=True, random_state=42)


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def average_simple_cv_results(evaluation_scorers, num_test_splits, hypparams_dict_list,
                              tr_result_matrix, vl_result_matrix, ts_result_matrix):
    cv_results = defaultdict(list)
    
    for hypparams_dict in hypparams_dict_list:
        cv_results['params'].append(hypparams_dict)
        for hypparam_name, hypparam_value in hypparams_dict.items():
            cv_results[f'param_{hypparam_name}'].append(hypparam_value)

    for scorer_idx, scorer_name in enumerate(evaluation_scorers.keys()):
        for round_idx in range(num_test_splits):
            cv_results[f"split{round_idx}_train_{scorer_name}"] = tr_result_matrix[scorer_idx, round_idx]
            cv_results[f"split{round_idx}_validation_{scorer_name}"] = vl_result_matrix[scorer_idx, round_idx]
            cv_results[f"split{round_idx}_test_{scorer_name}"] = ts_result_matrix[scorer_idx, round_idx]
        
        cv_results[f"mean_train_{scorer_name}"] = np.nanmean(tr_result_matrix[scorer_idx], axis=0).round(4)
        cv_results[f"std_train_{scorer_name}"] = np.nanstd(tr_result_matrix[scorer_idx], axis=0).round(4)
        
        cv_results[f"mean_validation_{scorer_name}"] = np.nanmean(vl_result_matrix[scorer_idx], axis=0).round(4)
        cv_results[f"std_validation_{scorer_name}"] = np.nanstd(vl_result_matrix[scorer_idx], axis=0).round(4)
        
        cv_results[f"mean_test_{scorer_name}"] = np.nanmean(ts_result_matrix[scorer_idx], axis=0).round(4)
        cv_results[f"std_test_{scorer_name}"] = np.nanstd(ts_result_matrix[scorer_idx], axis=0).round(4)
    return cv_results

def average_across_splits(split_results):
    final_results = split_results.copy()
    for metric_name, metric_values in split_results.items():
        if metric_name.startswith('mean') or metric_name.startswith('std'):
            final_results[metric_name] = np.nanmean(metric_values).round(3)
            if 'refit' in metric_name:
                final_results[f"std_{metric_name[5:]}"] = np.nanstd(metric_values).round(3)
    return final_results

def get_strategy(selection_metric, selection_phase, verbose_selection):
    if '_min_' in selection_metric:
        # e.g.: '90_accuracy_min_demographicParity'
        spl = selection_metric.split('_')
        percent = float(spl[0])
        utility_fun = spl[1]
        fairness_fun = spl[3]
        strategy = BestDDPOnUtilityPercentile(ddp_metric = fairness_fun,
                                              utility_metric = utility_fun,
                                              max_accuracy_percentile = percent,
                                              is_ddp_negated = True,
                                              phase = selection_phase,
                                              verbose = verbose_selection)
    else:
        strategy = SimpleBestMetricStrategy(evaluation_metric = selection_metric,
                                            greater_is_better = True,
                                            phase = selection_phase,
                                            verbose = verbose_selection)
                                            
    return strategy

def cv_results_name(data, include_sens, bias_mitigation, *, balance_classes=False, kernel=None):
    file_name = f"{data}"
    if include_sens: file_name += "_sensitive"
    if balance_classes: file_name += "_balancedClasses"
    if kernel is not None: file_name += f"_{kernel}Kernel"
    file_name += f"_{bias_mitigation if bias_mitigation is not None else 'noMitigation'}.pickle"
    return file_name