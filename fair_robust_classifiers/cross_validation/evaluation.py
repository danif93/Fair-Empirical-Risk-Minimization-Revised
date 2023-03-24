# ----- Standard Imports
import os
import pickle
from collections import defaultdict

# ----- Third Party Imports
from tqdm.auto import tqdm

# ----- Library Imports    
from fair_robust_classifiers.metrics.scorers import BaseScorer
from fair_robust_classifiers.cross_validation.cv_utils import get_strategy, average_across_splits


def evaluate_bias_mitigation(result_load_path,
                             result_file_name,
                             selection_metric,
                             evaluation_scorers,
                             selection_phase = 'validation',
                             verbose: int = 0, #0, 1, 2
                             ):
    # ----- Arguments validation
    assert isinstance(evaluation_scorers, dict) and \
           all([isinstance(scorer, BaseScorer) for scorer in evaluation_scorers.values()])
    if verbose not in [0,1,2]: verbose = 0
    
    # ----- Initialize hyperparameters selection strategy
    strategy = get_strategy(selection_metric, selection_phase, verbose>1)
        
    # ----- Load cross-validation results
    res_full_path = os.path.join(result_load_path, result_file_name)
    info_dict = pickle.load(open(res_full_path, "rb"))
    splits_results_list = info_dict['grid_search_info_list']
    
    # ----- Cycle across train/test splits and related validation results
    if verbose > 0:
        print(f"\nTraining/testing models with the best found hyperparameters configuration for {selection_metric} on {result_file_name} mitigation")
        split_cycle = tqdm(enumerate(splits_results_list))
    else:
        split_cycle = enumerate(splits_results_list)

    split_results = defaultdict(list)
    for split_idx, split_info_dict in split_cycle:
        # ----- Retrieve best cv results 
        cv_results = split_info_dict['cv_results']
        best_idx = strategy(cv_results)        
        best_params = cv_results['params'][best_idx]
        split_results['params'].append(best_params)
        
        # ----- Evaluate trained model
        for scorer_name in evaluation_scorers.keys():
            trn_avg_score = cv_results[f"mean_train_{scorer_name}"][best_idx]
            split_results[f"split{split_idx}_train_{scorer_name}"] = trn_avg_score
            split_results[f"mean_train_{scorer_name}"].append(trn_avg_score)
            trn_std_score = cv_results[f"std_train_{scorer_name}"][best_idx]
            split_results[f"std_train_{scorer_name}"].append(trn_std_score)
            
            vld_avg_score = cv_results[f"mean_validation_{scorer_name}"][best_idx]
            split_results[f"split{split_idx}_validation_{scorer_name}"] = vld_avg_score
            split_results[f"mean_validation_{scorer_name}"].append(vld_avg_score)
            vld_std_score = cv_results[f"std_validation_{scorer_name}"][best_idx]
            split_results[f"std_validation_{scorer_name}"].append(vld_std_score)
            
            tst_avg_score = cv_results[f"mean_test_{scorer_name}"][best_idx]
            split_results[f"split{split_idx}_test_{scorer_name}"] = tst_avg_score
            split_results[f"mean_test_{scorer_name}"].append(tst_avg_score)
            tst_std_score = cv_results[f"std_test_{scorer_name}"][best_idx]
            split_results[f"std_test_{scorer_name}"].append(tst_std_score)
            
    # ----- Average results across splits
    final_results = average_across_splits(split_results)
    
    store_name = f"{selection_metric}__{result_file_name}"
    file_path = os.path.join(result_load_path, store_name)
    pickle.dump(final_results, open(file_path, 'wb'))



