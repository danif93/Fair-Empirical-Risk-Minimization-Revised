# ----- Standard Imports
import os
import sys
import argparse

# ----- Third Party Imports
from sklearn.metrics import balanced_accuracy_score

# ----- Library Imports
if __name__ == "__main__":
    sys.path.append(os.path.join('..','..'))
    
from fair_robust_classifiers.metrics.scorers import SubgroupsMetricScorer, DDPMetricScorer, CounterfactualScorer
from fair_robust_classifiers.cross_validation.methods import gurobi_simple_cv_bias_mitigation
from fair_robust_classifiers.cross_validation.evaluation import evaluate_bias_mitigation
from fair_robust_classifiers.cross_validation.cv_utils import cv_results_name


def main():
    os.chdir(os.path.join("..", ".."))
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_test_splits',
        type = int,
        default = 10,
        help = "How many training rounds are run.")
    parser.add_argument(
        '--num_samples',
        type = int,
        default = 0,
        help = "How many samples are used for the outer train/test split.")
    parser.add_argument(
        '--bias_mitigation',
        type = str,
        choices = ['linearDP', 'linearEOpPos', 'linearEOpNeg',
                   'invertedHingesDPfull', 'invertedHingesDP', 'invertedHingesDP0', 'invertedHingesDP1',
                   'invertedHingesEOpPosfull', 'invertedHingesEOpPos', 'invertedHingesEOpPos0', 'invertedHingesEOpPos1',
                   'invertedHingesEOpNegfull', 'invertedHingesEOpNeg', 'invertedHingesEOpNeg0', 'invertedHingesEOpNeg1'],
        help = "Which bias mitigation algortihm to deploy.",)
    parser.add_argument(
        '--kernel',
        type = str,
        choices = ['linear', 'sigmoidal', 'gaussian'],
        help = 'Which kernel to use.')
    args = parser.parse_args()
            
    cv_scorers = {
        # utility scorers
        "balancedAccuracy": SubgroupsMetricScorer(balanced_accuracy_score, need_class_predictions=True),

        # fairness scorer
        'demographicParity': DDPMetricScorer(),
        'equalOpportunityPos': DDPMetricScorer(evaluation_class=1),
        'equalOpportunityNeg': DDPMetricScorer(evaluation_class=-1),
        "counterfactual": CounterfactualScorer(),
        "counterfactualPos": CounterfactualScorer(evaluation_class=1),
        "counterfactualNeg": CounterfactualScorer(evaluation_class=-1),
    }
    cv_scorers_red = {k:v for k,v in cv_scorers.items() if 'counterfactual' not in k}
    
    sel_metrics = [
        "balancedAccuracy",

        "demographicParity",
        "equalOpportunityNeg",
        "equalOpportunityPos",
        "counterfactual",
        "counterfactualPos",
        "counterfactualNeg",

        "90_balancedAccuracy_min_demographicParity",
        "90_balancedAccuracy_min_equalOpportunityNeg",
        "90_balancedAccuracy_min_equalOpportunityPos",
        "90_balancedAccuracy_min_counterfactual",
        "90_balancedAccuracy_min_counterfactualPos",
        "90_balancedAccuracy_min_counterfactualNeg",
    ]
    sel_metrics_red = [sm for sm in sel_metrics if 'counterfactual' not in sm]

    load_path = os.path.join('results', 'gurobiSVC', f'grid_search_results_simple')

    balance_classes = False
    
    for data_str, label_str, sensitive_str in [
        ('arrhythmia', 'hasArrhythmia','sex'),
        
        #('adult', 'grossIncomeGEQ50k','race'),
        ('adult', 'grossIncomeGEQ50k','sex'), # <-
        #('adult', 'grossIncomeGEQ50k','nativeCountry'),

        ('credit', 'NoDefaultNextMonth', 'Age'),

        #('drug', 'heroin', 'gender'),
        #('drug', 'heroin', 'ethnicity'),
        ('drug', 'amphetamines', 'gender'), # <-
        #('drug', 'amphetamines', 'ethnicity'),
        
        ('germanSex', 'creditRisk', 'sex'),
        
        #('compas', 'twoYearRecid','sex'),
        ('compas', 'twoYearRecid','race'), # <-
        
        ('taiwan', 'defaultNextMonth', 'sex'),
    ]:
        for include_sens in [True, False]:
            print(f"\nCross-validating and training over {data_str} with {args.bias_mitigation} mitigation and kernel {args.kernel}")
            print(f"\t{'Including' if include_sens else 'Excluding'} the sensitive attribute within model training\n")

            file_name = cv_results_name(data_str, include_sens, args.bias_mitigation,
                                        balance_classes=balance_classes, kernel=args.kernel)
            
            gurobi_simple_cv_bias_mitigation(data_str, label_str, sensitive_str,
                                             evaluation_scorers = cv_scorers if include_sens else cv_scorers_red,
                                             kernel = args.kernel,
                                             bias_mitigation = args.bias_mitigation,
                                             num_test_splits = args.train_test_splits,
                                             num_samples = args.num_samples,
                                             balance_classes = False,
                                             include_sensitive = include_sens,
                                             train_percentage = .7)
            
            for sel_metric in (sel_metrics if include_sens else sel_metrics_red):
                evaluate_bias_mitigation(selection_metric = sel_metric,
                                         evaluation_scorers = cv_scorers if include_sens else cv_scorers_red,
                                         result_load_path = load_path,
                                         result_file_name = file_name,
                                         selection_phase = 'validation',
                                         verbose = 1)


if __name__ == "__main__":
    main()
    