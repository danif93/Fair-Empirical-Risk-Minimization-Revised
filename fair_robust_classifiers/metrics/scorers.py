# ----- Standard Imports
from copy import deepcopy

# ----- Third-Party Imports
import numpy as np

# ----- Library Imports


# ---------------------------------------
# --- Cross-Validation Evaluation Scorers
# ---------------------------------------

class BaseScorer:
    def __init__(self,
                 evaluation_class = None,
                 sensitive_group = None,
                 ) -> None:
        self.eval_class = evaluation_class
        self.sens_group = sensitive_group

    def __call__(self, estimator, X, y):
        pass
    
    def retrieve_data_mask(self,
                           mask_length: int,
                           y = None,
                           s = None,
                           ):
        test_mask = np.full(mask_length, True)
        if  self.eval_class is not None:
            test_mask &= y ==  self.eval_class
        if  self.sens_group is not None:
            test_mask &= s ==  self.sens_group
        return test_mask
    
    
class SubgroupsMetricScorer(BaseScorer):
    def __init__(self,
                 evaluation_metric,
                 evaluation_class = None,
                 sensitive_group = None,
                 need_class_predictions = False,
                 inverse_metric = False,
                ):
        
        super().__init__(evaluation_class=evaluation_class, sensitive_group=sensitive_group)
        self.eval_metric = evaluation_metric
        self.need_classes = need_class_predictions
        self.inverse_metric = inverse_metric
    
    def __call__(self, estimator, X, y):
        # each element is the pair (standard features, sensitive feature)
        if all([isinstance(pair, tuple) and len(pair)==2 for pair in X]): 
            features, sens_feature = map(lambda xf: np.stack(xf), zip(*X))
            
        # ----- Retrieve the test samples
        test_mask = self.retrieve_data_mask(features.shape[0], y=y, s=sens_feature)
        if not test_mask.any():
            # raise RuntimeError(f"No samples with sensitive {self.sens_group} and class {self.eval_class} have been found in the batch")
            return np.nan    
                
        test_samples = features[test_mask]
        test_labels = y[test_mask]
        
        # ----- Compute model predictions
        if self.need_classes:
            y_pred = estimator.predict(test_samples)
        else:
            y_pred = estimator.decision_function(test_samples)
                
        score = self.eval_metric(test_labels, y_pred)

        if self.inverse_metric:
            score = -score
            
        return round(score, 3)
    
    
class DDPMetricScorer(BaseScorer):
    def __init__(self,
                 evaluation_metric = None,
                 sensitive_group = None,
                 evaluation_class = None,
                 need_class_predictions = False,
                ):
        super().__init__(evaluation_class=evaluation_class, sensitive_group=None)
        self.eval_metric = evaluation_metric
        self.eval_group = sensitive_group
        self.need_classes = need_class_predictions
    
    def __call__(self, estimator, X, y):
        if not all([isinstance(pair, tuple) and len(pair)==2 for pair in X]):
            raise ValueError("Argument X must be a list of pairs where the first element is the feed features while the second one is the sensitive feature vector.")
        features, sens_feature = map(lambda xf: np.stack(xf), zip(*X))
        
        # ----- Retrieve the test samples
        class_mask = self.retrieve_data_mask(features.shape[0], y=y)
        if not class_mask.any():
            # raise RuntimeError("There are no samples within the evaluated class")
            return np.nan
        
        test_samples = features[class_mask]
        test_sensitive = sens_feature[class_mask]
        test_labels = y[class_mask]
            
        if self.eval_group is None:
            unique_groups = np.unique(test_sensitive)
            if len(unique_groups) > 2:
                raise RuntimeError("Since a sensitive population is not specified, the sensitive vector must be binary.")
            chosen_sens = unique_groups[0]
        else:
            chosen_sens = self.eval_group
            
        evl_group_mask = test_sensitive == chosen_sens
        oth_group_mask = ~evl_group_mask
        if evl_group_mask.all() or oth_group_mask.all():
            # raise RuntimeError("Either all or none of the samples belong to the evaluated group")
            return np.nan
        
        # ----- Compute model predictions
        if self.need_classes:
            y_pred = estimator.predict(test_samples)
        else:
            y_pred = estimator.decision_function(test_samples)
            
        if self.eval_metric is not None:
            score_evl_group = self.eval_metric(test_labels[evl_group_mask], y_pred[evl_group_mask])
            score_oth_group = self.eval_metric(test_labels[oth_group_mask], y_pred[oth_group_mask])
        else:
            score_evl_group = y_pred[evl_group_mask].mean()
            score_oth_group = y_pred[oth_group_mask].mean()
            
        score = np.abs(score_evl_group - score_oth_group)
        score = -score
        
        return round(score, 3)


class CounterfactualScorer(BaseScorer):
    def __init__(self,
                 evaluation_class = None,
                 need_class_predictions = False,
                 ):
        super().__init__(evaluation_class=evaluation_class, sensitive_group=None)        
        self.need_classes = need_class_predictions
    
    def __call__(self, estimator, X, y):
        # ----- argument checking
        if not all([isinstance(pair, tuple) and len(pair)==2 for pair in X]):
            raise ValueError("Argument X must be a list of pairs where the first element is the feed features while the second one is the sensitive feature vector.")
        features, sens_feature = map(lambda xf: np.stack(xf), zip(*X))
        assert all([sv in [0,1] for sv in sens_feature]) #TOFIX: assumes sensitive in [0,1]
        assert all([sv in [0,1] for sv in features[:,-1]]) #TOFIX: assumes sensitive in [0,1] and sensitive attribute in the last column

        # ----- Retrieve the test samples
        class_mask = self.retrieve_data_mask(features.shape[0], y=y)
        if not class_mask.any():
            # raise RuntimeError("There are no samples within the evaluated class")
            return np.nan
        
        test_samples = features[class_mask]

        # ----- Compute model predictions
        count_samples = deepcopy(test_samples)
        count_samples[:,-1] = 1 - count_samples[:,-1]
        
        if self.need_classes:
            y_pred_real = estimator.predict(test_samples)
            y_pred_count = estimator.predict(count_samples)
        else:
            y_pred_real = estimator.decision_function(test_samples)
            y_pred_count = estimator.decision_function(count_samples)
                
        score = np.abs(y_pred_real - y_pred_count).mean()
        score = -score
        
        return round(score, 3)