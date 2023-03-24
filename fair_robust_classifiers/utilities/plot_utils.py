# ----- Standard Imports
import pickle
import os
import warnings

# ----- Third Party Imports
import pandas as pd
import matplotlib.pyplot as plt

# ----- Library Imports
from fair_robust_classifiers.cross_validation.cv_utils import cv_results_name


# -------------------------------------
# --- Cross-Validation Evaluation Table
# -------------------------------------

def make_fairness_results_table(data,
                                selection_metrics,
                                mitigation_methods,
                                evaluation_metrics,
                                include_sensitive = True,
                                balance_classes = False,
                                kernel = None,
                                phase = 'test',
                                folder_path = "grid_search_results",
                               ):
    # ----- Arguments validation
    assert phase in ['train','test','validation','refitTrain','refitTest']
    
    # ----- Auxiliary structures
    mitigation_metric_map = {
        'noMitigation': ["demographicParity", "equalOpportunityNeg", "equalOpportunityPos",
                         "counterfactual", 'counterfactualPos', 'counterfactualNeg'],
            # Fair SVC
        "linearDP": ["demographicParity",  "counterfactual"],
        "linearEOPpos": ["equalOpportunityPos", 'counterfactualPos'],
        "linearEOPneg": ["equalOpportunityNeg", 'counterfactualNeg'],
        "invertedHingesDP": ["demographicParity", "counterfactual"],
        "invertedHingesEOPpos": ["equalOpportunityPos", 'counterfactualPos'],
        "invertedHingesEOPneg": ["equalOpportunityNeg", 'counterfactualNeg'],
    }
    global_metrics = ['accuracy', 'balancedAccuracy']
    
    # ----- Result table initialization
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    pd.options.display.max_colwidth = 120
    pd_res = pd.DataFrame(columns=evaluation_metrics)
    
    # ----- Cycle through the hyperparameters selection strategies
    for sel_metric in selection_metrics:
        if '_min_' in sel_metric:
            aux_sel = sel_metric.split('_')[3]
        else:
            aux_sel = sel_metric
            
        # ----- Cycle through the evaluated bias mitigation constraints
        for bias_mit in mitigation_methods:
            bias_metrics = mitigation_metric_map[bias_mit] + global_metrics
            if not aux_sel in bias_metrics: continue
                
            # ----- Load average test results
            file_name = f"{sel_metric}__{cv_results_name(data, include_sensitive, bias_mit, balance_classes=balance_classes, kernel=kernel)}"
            cv_results = pickle.load(open(os.path.join(folder_path, file_name), 'rb'))
            
            row_idx = f"{sel_metric}__{bias_mit}"
            # ----- Cycle through the stored results and add to table
            for full_eval_metric, eval_res in cv_results.items():
                if f'mean_{phase}' in full_eval_metric or full_eval_metric=='params':
                    if full_eval_metric == 'params':
                        cv_eval_mtr = 'params'
                        str_val = [{k:v for k,v in d.items()}
                                   for d in eval_res]
                    else:
                        cv_eval_mtr = full_eval_metric.split('_')[2]
                        str_val = f"{eval_res}±{cv_results[f'std_{phase}_{cv_eval_mtr}']}"

                    if cv_eval_mtr in evaluation_metrics and \
                       cv_eval_mtr in bias_metrics and \
                       (cv_eval_mtr == aux_sel or aux_sel in global_metrics or cv_eval_mtr in global_metrics):
                        pd_res.loc[row_idx, cv_eval_mtr] = str_val
    return pd_res.fillna('')


# -------------------------------------------
# --- Cross-Validation Parameters Scatterplot
# -------------------------------------------

def normalized_accuracy_fairness_plot_cum(plot_dict):
    aux_dict = {
        'adult':'v','arrhythmia':'^','compas':'o',
        'credit':'s','drug':'D','germanSex':'+','taiwan':'*',
        'noMiti':'#DB4437', 'linear':'#4285F4', 'invert':'#0F9D58',
        'noMiti_lbl':'no constr.', 'linear_lbl':'FERM', 'invert_lbl':'R-FERM',
        'balancedAccuracy':'BA', 'demographicParity':'DP',
        'counterfactual':'CF',
        'equalOpportunityPos':'EO', 'equalOpportunityNeg':'EO',
    }

    def normalize_01(data_fr):
        df_min = data_fr.min()
        df_max = data_fr.max()
        data_fr -= df_min
        if df_max != df_min:
            data_fr /= df_max - df_min
        return data_fr

    rows = len(plot_dict['settings'])
    cols = len(plot_dict['settings'][0])
    fig, axs = plt.subplots(rows,cols, figsize=(12*rows, 1.2*cols), squeeze=False, sharex=True, sharey=True)

    for out_idx, outer in enumerate(plot_dict['settings']):
        for in_idx, inner in enumerate(outer):
            for data, eo_class in plot_dict['datasets'].items():
                fair_fn = f"{inner['fair_fn']}{eo_class if inner['fair_fn']=='equalOpportunity' else ''}"
                selection = f"{inner['percent']}_{inner['util_fn']}_min_{fair_fn}"
                df = pd.DataFrame(columns = [inner['util_fn'], fair_fn])
                for mitigation in ['noMitigation', f"linear{inner['mitig']}", f"invertedHinges{inner['mitig']}"]:
                    mitigation = f"{mitigation}{eo_class if mitigation!='noMitigation' and inner['mitig']=='EOp' else ''}"
                    setting = f"{data}_{'sensitive_' if inner['sensitive'] else ''}{mitigation}"
                    res = pickle.load(open(os.path.join(plot_dict['file_path'], f"{selection}__{setting}.pickle"), 'rb'))
                    df.loc[mitigation, inner['util_fn']] = res[f"mean_{plot_dict['phase']}_{inner['util_fn']}"]
                    df.loc[mitigation, fair_fn] = res[f"mean_{plot_dict['phase']}_{fair_fn}"]
                df = df.apply(normalize_01)

                for idx, values in df.iterrows():
                    axs[out_idx][in_idx].plot(1-values[fair_fn], 1-values[inner['util_fn']],
                                              aux_dict[data], markerfacecolor='none', markersize=12,
                                              color=aux_dict[idx[:6]])
                    
            axs[out_idx][in_idx].grid()
            spl_title = selection.split('_')
            axs[-1][in_idx].set_xlabel(aux_dict[spl_title[-1]], size=15)
            axs[out_idx][0].set_ylabel(f"{aux_dict[inner['util_fn']]} Error", size=15)
            
    for d in plot_dict['datasets']:        
        axs[-1][-1].plot([],aux_dict[d], markerfacecolor='none', color='k', label=d if not d.startswith('germ') else 'german')
    for method in ['noMiti','linear','invert']:
        axs[-1][-1].plot([], 's', color=aux_dict[method], label=aux_dict[f"{method}_lbl"])
    axs[-1][-1].legend(bbox_to_anchor=(1.05, 1.05), loc='upper left', borderaxespad=0, fontsize=14, markerscale=1.5, handletextpad=0)
    fig.tight_layout()
