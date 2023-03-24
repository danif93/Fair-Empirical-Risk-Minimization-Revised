# ----- Standard Imports
import os
import pickle

# ----- Third Party Imports
import numpy as np
from tqdm.auto import tqdm

# ----- Library Imports
from fair_robust_classifiers.models.gurobi_svm import GurobiSVC
from fair_robust_classifiers.models.model_utils import OptimizationError, EmptyVectorError
from fair_robust_classifiers.cross_validation.cv_utils import (SCALER,
                                                               average_simple_cv_results,
                                                               product_dict,
                                                               cv_results_name)
from fair_robust_classifiers.datasets.data_utils import load_data    


# ---------------------------
# --- Custom Cross-validation
# ---------------------------

def gurobi_simple_cv_bias_mitigation(data, label, sensitive, 
                                     evaluation_scorers,
                                     kernel = None,
                                     bias_mitigation = None,
                                     num_test_splits = 10,
                                     num_samples = -1,
                                     balance_classes = False,
                                     include_sensitive = True,
                                     train_percentage = .7,
                                     ):
    # ----- Set fixed and cv model parameters
    _, fixed_params, hypparams_grid = \
        GurobiSVC.get_model_name_and_parameters(soft_margin = 'cv',
                                                balance_classes = balance_classes,
                                                kernel = kernel,
                                                gamma = 'cv',
                                                bias_mitigation = bias_mitigation,
                                                fairness_param = 'cv')
    hypparams_dict_list = list(product_dict(**hypparams_grid))
    
    # ----- Load data
    X, y, s = load_data(data, label, sensitive,
                        merge_sensitive = include_sensitive,
                        num_samples = num_samples)
    num_samples = len(X)
    tr_end_idx = int(num_samples * train_percentage)
    vl_end_idx = tr_end_idx + (num_samples - tr_end_idx)//2
    data_scaler = SCALER
    
    # ----- Initialize results matrices
    tr_result_matrix = np.empty((len(evaluation_scorers),
                                 num_test_splits,
                                 len(hypparams_dict_list)),
                                dtype = float)
    vl_result_matrix = np.empty((len(evaluation_scorers),
                                 num_test_splits,
                                 len(hypparams_dict_list)),
                                dtype = float)
    ts_result_matrix = np.empty((len(evaluation_scorers),
                                 num_test_splits,
                                 len(hypparams_dict_list)),
                                dtype = float)
    
    # ----- Cycle across train/validation/test splits
    for round_idx in range(num_test_splits):
        print(f"\nRunning simple cross-validation [{round_idx+1}/{num_test_splits}] "\
              f"for each one of the {len(hypparams_dict_list)} candidates.")
        
        # ----- Retrieve train/valid/test indexes and samples
        rng = np.random.default_rng(round_idx)
        perm_idxs = rng.permutation(num_samples)
        tr_idxs, vl_idxs, ts_idxs = \
            perm_idxs[:tr_end_idx], perm_idxs[tr_end_idx:vl_end_idx], perm_idxs[vl_end_idx:]
        
        X_train, X_valid, X_test = X[tr_idxs], X[vl_idxs], X[ts_idxs]
        X_train = data_scaler.fit_transform(X_train)
        X_valid = data_scaler.transform(X_valid)
        X_test = data_scaler.transform(X_test)
        s_train, s_valid, s_test = s[tr_idxs], s[vl_idxs], s[ts_idxs]
        y_train, y_valid, y_test = y[tr_idxs], y[vl_idxs], y[ts_idxs]
        
        tr_feed = list(zip(X_train, s_train))
        vl_feed = list(zip(X_valid, s_valid))
        ts_feed = list(zip(X_test, s_test))
    
        # ----- Cycle across hyper-parameters
        for hypparams_idx, hypparams_dict in tqdm(enumerate(hypparams_dict_list)):
            # ----- Initialize model
            model = GurobiSVC(**fixed_params, **hypparams_dict)

            # ----- Train model
            try:
                trained = True
                model.fit(tr_feed, y_train)
                
            except OptimizationError:
                print(f"Gurobi SVC [{model.get_params()}] has no solution if trained on {data} [trn/vld/tst split {round_idx}].")
                trained = False
            except EmptyVectorError:
                print(f"Data {data} [test split {round_idx}] does not contain enough sensitive samples "\
                      f"[training Gurobi SVC with configuration {model.get_params()}].")
                trained = False

            # ----- Cycle across scores and evaluate on train, validation and test sets
            for scorer_idx, scorer in enumerate(evaluation_scorers.values()):
                if trained:
                    train_score = scorer(model, tr_feed, y_train)
                    tr_result_matrix[scorer_idx, round_idx, hypparams_idx] = train_score

                    valid_score = scorer(model, vl_feed, y_valid)
                    vl_result_matrix[scorer_idx, round_idx, hypparams_idx] = valid_score

                    test_score = scorer(model, ts_feed, y_test)
                    ts_result_matrix[scorer_idx, round_idx, hypparams_idx] = test_score

                else:
                    tr_result_matrix[scorer_idx, round_idx, hypparams_idx] = np.nan
                    vl_result_matrix[scorer_idx, round_idx, hypparams_idx] = np.nan
                    ts_result_matrix[scorer_idx, round_idx, hypparams_idx] = np.nan
    
    # ----- Average and store hyperparameters results
    cv_results = average_simple_cv_results(evaluation_scorers, num_test_splits, hypparams_dict_list,
                                           tr_result_matrix, vl_result_matrix, ts_result_matrix)
    
    file_name = cv_results_name(data, include_sensitive, bias_mitigation,
                                balance_classes=balance_classes, kernel=kernel)
    folder_path = os.path.join('results', 'gurobiSVC', 'grid_search_results_simple')
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, file_name)
    pickle.dump({'grid_search_info_list': [{'cv_results': cv_results}],
                 'samples_number': num_samples},
                open(file_path, "wb"))
