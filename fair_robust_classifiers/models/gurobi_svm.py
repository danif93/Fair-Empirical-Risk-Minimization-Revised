# ----- Standard Imports

# ----- Third Party Imports
import numpy as np
from sklearn import base
import gurobipy as gp
from gurobipy import GRB, quicksum, GurobiError

# ----- Library Imports
from fair_robust_classifiers.models.model_utils import OptimizationError, EmptyVectorError, KERNEL_MAP


# --------------
# --- Gurobi SVM
# --------------

class GurobiSVC(base.BaseEstimator):
    def __init__(self,
                 c = 1.,
                 kernel = None, # 'linear', 'gaussian'
                 gamma = 'scale', # 'auto' or float value
                 #degree = 3,
                 gurobi_model_params = None,
                 balanced_class_weights = False,
                 bias_mitigation = None,
                 fairness_eps = 0.,
                 verbose = True,
                ):
        #super().__init__()
        self.c = c
        self.kernel = kernel
        self.gamma = gamma
        #self.degree = degree
        self.gurobi_model_params = gurobi_model_params
        self.balanced_class_weights = balanced_class_weights
        self.bias_mitigation = bias_mitigation
        self.fairness_eps = fairness_eps
        self.verbose = verbose

    @staticmethod
    def _remap_labels(y, positive_class_val):
        positives_mask = y == positive_class_val
        remap_y = np.empty_like(y)
        remap_y[positives_mask] = 1
        remap_y[~positives_mask] = -1
        return remap_y
    
    @staticmethod
    def _apply_fairness_constraint(constr_name, epsilon,
                                   data, sensitives, targets,
                                   gurobi_model, variables,
                                   kernel = None):
        # ----- Validate arguments
        if constr_name not in ['linearLossDP', 'linearDP', 'linearEOpPos', 'linearEOpNeg', 'linearEOd',
                               'invertedHingesDPfull', 'invertedHingesDP0', 'invertedHingesDP1',
                               'invertedHingesEOpPosfull', 'invertedHingesEOpPos0', 'invertedHingesEOpPos1',
                               'invertedHingesEOpNegfull', 'invertedHingesEOpNeg0', 'invertedHingesEOpNeg1']:
            raise ValueError(f"Invalid 'fairness_mitigation' argument; received: {constr_name}")
        if epsilon < 0:
            raise ValueError(f"Epsilon parameter in the fairness constraint must be >= 0; received: {epsilon}")
        if np.unique(targets).tolist() != [-1,1]:
            raise ValueError("Targets vector must contains only -1 and 1.")
            
        # ----- Set constraints parameters
        n_samples, n_features = data.shape[0], data.shape[1]
        if kernel is None:
            m_w = variables['w']
        else:
            m_alpha = variables['alpha']
        m_b, m_xi = variables['b'], variables['xi']
        
        sens_groups = np.unique(sensitives) # TODO: implement OvA constraint for dealing with the exception below
        if len(sens_groups) != 2: 
            raise ValueError(f"Only binary sensitive attributes are currently supported; the found unique values of 's' are {sens_groups}")
        gr0_msk = sensitives == sens_groups[0]
        gr1_msk = ~gr0_msk
        if gr0_msk.all() or gr1_msk.all():
            raise EmptyVectorError("Either one of the two vectors containing a sensitive group is empty")
            
        # ----- Set constraints
        
        # a) linear DP, Equal Opp[+,-] on model outputs: 
        #     w * (\sum_{i\in{0,[+|-]}}{x_i}/n_{0,[+|-]} - \sum_{i\in{1,[+|-]}}{x_i}/n_{1,[+|-]}) == 0
        if constr_name == 'linearDP' or constr_name.startswith('linearEOp'):
            if constr_name == 'linearDP':
                lbl_msk = np.full(n_samples, True)
            else:
                lbl_msk = targets == (1 if constr_name == 'linearEOpPos' else -1)
            
            gr0_lbl_msk = gr0_msk & lbl_msk
            gr1_lbl_msk = gr1_msk & lbl_msk
            if (not gr0_lbl_msk.any()) or (not gr1_lbl_msk.any()):
                raise EmptyVectorError("Either one of the two vectors containing a sensitive group is empty")
            
            avg_grfeatures_dist = np.mean(data[gr0_lbl_msk], axis=0) - np.mean(data[gr1_lbl_msk], axis=0)

            if kernel is None:
                fair_constr = quicksum(m_w[j] * avg_grfeatures_dist[j] for j in range(n_features)) == 0
            else:
                krnl_avg_dist = kernel(data, avg_grfeatures_dist.reshape(1,-1)).squeeze(1)
                fair_constr = quicksum(m_alpha[i] * krnl_avg_dist[i] for i in range(n_samples)) == 0
            gurobi_model.addConstr(fair_constr, name=constr_name)
            
        # b) inverted hinges DP on model outputs:
        #      w * x_i + b >= 1 - ro_i; ro_i >= 0
        #      w * x_i + b <= -eta_i;   eta_i <= 1
        #      \sum_{i\in{0}}{ro_i}/n_0 - \sum_{i\in{1}}{eta_i}/n_1 <= \epsilon
        #      or:
        #      \sum_{i\in{1}}{ro_i}/n_1 - \sum_{i\in{0}}{eta_i}/n_0 <= \epsilon
        elif constr_name.startswith('invertedHinges'):
            if constr_name.startswith('invertedHingesDP'):
                # use all the samples
                smpl_idxs = list(range(n_samples))
                trgt = np.full(n_samples, 1)
            else: # constr_name.startswith('invertedHingesEOp')
                # use only the selected class samples
                lbl_msk = targets == (1 if constr_name.startswith('invertedHingesEOpPos') else -1)
                smpl_idxs = np.where(lbl_msk)[0].tolist()
                trgt = targets[smpl_idxs]
                gr0_msk = sensitives[smpl_idxs] == sens_groups[0]
                gr1_msk = ~gr0_msk
                if gr0_msk.all() or gr1_msk.all():
                    raise EmptyVectorError("Either one of the two vectors containing a sensitive group is empty")
            
            # select the constraint samples
            cnstr_data = data[smpl_idxs]

            # add new ro and eta variables
            n_constr = len(smpl_idxs)
            ro = gurobi_model.addVars(n_constr, lb=0, ub=GRB.INFINITY)
            eta = gurobi_model.addVars(n_constr, lb=-GRB.INFINITY, ub=1)
            
            # local constraints
            if kernel is None:
                lcl_ro = (trgt[i] * (quicksum(m_w[j] * cnstr_data[i,j] for j in range(n_features)) + m_b) >= 1 - ro[i]
                          for i in range(n_constr))
                lcl_eta = (trgt[i] * (quicksum(m_w[j] * cnstr_data[i,j] for j in range(n_features)) + m_b) <= -eta[i]
                           for i in range(n_constr))
            else:
                kernel_mtx = kernel(data, cnstr_data)
                lcl_ro = (trgt[i] * (quicksum(m_alpha[j] * kernel_mtx[j,i] for j in range(n_samples)) + m_b) >= 1 - ro[i]
                          for i in range(n_constr))
                lcl_eta = (trgt[i] * (quicksum(m_alpha[j] * kernel_mtx[j,i] for j in range(n_samples)) + m_b) <= -eta[i]
                           for i in range(n_constr))
            gurobi_model.addConstrs(lcl_ro, name=f"{constr_name}_localRoConstrs")
            gurobi_model.addConstrs(lcl_eta, name=f"{constr_name}_localEtaConstrs")
            
            # global constraints
            gr0_idxs = np.where(gr0_msk)[0].tolist()
            gr1_idxs = np.where(gr1_msk)[0].tolist()

            if constr_name.endswith('0') or constr_name.endswith('full'):
                diff_gr0_constr = quicksum(ro[i_0] for i_0 in gr0_idxs)/len(gr0_idxs) - \
                                  quicksum(eta[i_1] for i_1 in gr1_idxs)/len(gr1_idxs) <= epsilon
                gurobi_model.addConstr(diff_gr0_constr, name=f"{constr_name}_globalAvgRoEtaDiff0")
            if constr_name.endswith('1') or constr_name.endswith('full'):
                diff_gr1_constr = quicksum(ro[i_1] for i_1 in gr1_idxs)/len(gr1_idxs) - \
                                  quicksum(eta[i_0] for i_0 in gr0_idxs)/len(gr0_idxs) <= epsilon
                gurobi_model.addConstr(diff_gr1_constr, name=f"{constr_name}_globalAvgRoEtaDiff1")
            
        return gurobi_model
                
    def fit(self, X, y):
        # ----- model parameters checking
        if self.c < 0:
            raise ValueError(f"Penalty term must be positive; got {self.c}")
        if self.kernel is not None and self.kernel not in ('linear','sigmoidal','gaussian',
                                                           #'polynomial',
                                                          ):
            raise ValueError(f"Kernel must be None or one of ['linear','sigmoidal','gaussian']; got {self.kernel}")
        
        # ----- set training parameters
        # each element might be the pair (standard features, sensitive feature)
        if all([isinstance(pair, tuple) and len(pair)==2 for pair in X]):
            X, s = map(lambda xf: np.stack(xf), zip(*X))
        
        num_samples, self.n_features_in_ = X.shape[0], X.shape[1]
        self.classes_, inv_y_map, val_counts = np.unique(y, return_inverse=True, return_counts=True)
        if len(self.classes_) != 2:
            raise ValueError("For now, the implemented GurobiSVM support only binary classification problems")
            
        if self.balanced_class_weights:
            cls_weights = num_samples / (len(self.classes_) * val_counts)
            tr_c = self.c * np.array([cls_weights[i] for i in inv_y_map])
        else:
            tr_c = np.full(num_samples, self.c)
        
        rm_y = self._remap_labels(y, positive_class_val=self.classes_[1])
        
        if self.kernel is not None:
            krnl = KERNEL_MAP[self.kernel]
            self.kernel_class_ = krnl(gamma = self.gamma,
                                      #degree = self.degree,
                                      # add here the parameters for other kernels when implemented
                                     ) 

        # ----- initialize optimization model
        model = gp.Model()
        model.setParam('OutputFlag', int(self.verbose))
        if self.gurobi_model_params is not None:
            assert isinstance(self.gurobi_model_params, dict), \
                "'gurobi_model_params' must be a dictionary containing (name, value) model parameters"
            for param_name, param_value in self.gurobi_model_params.items():
                model.setParam(param_name, param_value)
        optim_vars = {}
                
        if self.kernel is None:
            # ----- add optimization variables
            w = model.addVars(self.n_features_in_, lb=-GRB.INFINITY, ub=GRB.INFINITY)
            b = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
            xi = model.addVars(num_samples, lb=0, ub=GRB.INFINITY)
            optim_vars['w'] = w
            optim_vars['b'] = b
            optim_vars['xi'] = xi
            
            # ----- define optimization objective
                # 1/2 * ||w||_2^2 + C_{y_i} * \sum_i{xi_i}
            obj = quicksum(w[j]**2 for j in range(self.n_features_in_)) / 2 + \
                  quicksum(tr_c[i] * xi[i] for i in range(num_samples))
            model.setObjective(obj, GRB.MINIMIZE)
            
            # ----- define soft-margin constraints
                # y_i * (x_i * w^T + b) >= 1 - xi_i  \forall i \in dataset-indexes
            sm_constrs = (rm_y[i] * (quicksum(w[j] * X[i][j] for j in range(self.n_features_in_)) + b) \
                          >= 1 - xi[i] for i in range(num_samples))
            model.addConstrs(sm_constrs, name="soft-margin")
        
        else:
            # ----- add optimization variables
            alpha = model.addVars(num_samples, lb=-GRB.INFINITY, ub=GRB.INFINITY)
            b = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
            xi = model.addVars(num_samples, lb=0, ub=GRB.INFINITY)
            optim_vars['alpha'] = alpha
            optim_vars['b'] = b
            optim_vars['xi'] = xi 
            
            krnl_mtx = self.kernel_class_(X)
            
            # ----- define optimization objective:
                # 1/2 * \sum_i{\sum_j{alpha_i * alpha_j * K(x_i,x_j)}} + C_{y_i} * \sum_i{xi_i}
            obj = quicksum(quicksum(alpha[i] * alpha[j] * krnl_mtx[i,j] 
                                    for j in range(num_samples))
                           for i in range(num_samples)) / 2 + \
                  quicksum(tr_c[i] * xi[i] for i in range(num_samples))
            model.setObjective(obj, GRB.MINIMIZE)

            # ----- define soft-margin constraints
                # y_i * (\sum_j{alpha_j * K(x_i,x_j)} + b) >= 1 - xi_i  \forall i \in dataset-indexes
            sm_constrs = (rm_y[i] * (quicksum(alpha[j] * krnl_mtx[i,j] for j in range(num_samples)) + b) \
                          >= 1 - xi[i] for i in range(num_samples))
            model.addConstrs(sm_constrs, name="kernel soft-margin")
        
        # fairness constraints
        if self.bias_mitigation is not None:
            model = self._apply_fairness_constraint(constr_name = self.bias_mitigation,
                                                    epsilon = self.fairness_eps,
                                                    data=X, sensitives=s, targets=rm_y,
                                                    gurobi_model = model,
                                                    variables = optim_vars,
                                                    kernel = getattr(self, 'kernel_class_', None))
        
        # ----- optimize
        try:
            model.optimize()
        except GurobiError as ge:
            raise OptimizationError(ge)

        if model.Status != 2:
            raise OptimizationError("No optimal solution has been found.")
        
        # ----- store model coefficient
        if self.kernel is None:
            self.coef_ = np.array([w[i].x for i in w.keys()])
            self.intercept_ = np.array(b.x)
        else:
            self.dual_coef_ = np.array([alpha[i].x for i in alpha.keys()]).round(4)
            self.intercept_ = np.array(b.x)
            s_v_msk = self.dual_coef_ != 0
            if not s_v_msk.any():
                raise EmptyVectorError('No support vectors have been found.')
            self.dual_coef_ = self.dual_coef_[s_v_msk, None]
            self.support_vectors_ = X[s_v_msk]
            
        return self
    
    def decision_function(self, X):
        # each element is the pair (standard features, sensitive feature)
        if all([isinstance(pair, tuple) and len(pair)==2 for pair in X]):
            X, _ = map(lambda xf: np.stack(xf), zip(*X))
        
        if self.kernel is None:
            raw = X.dot(self.coef_) + self.intercept_
        else:
            krnl_mtx = self.kernel_class_(self.support_vectors_, X)
            raw = (self.dual_coef_ * krnl_mtx).sum(axis=0) + self.intercept_
        return raw

    def predict(self, X):
        pred = np.where(self.decision_function(X) >= 0, self.classes_[1], self.classes_[0])
        return pred
    
    @staticmethod
    def get_model_name_and_parameters(soft_margin,
                                      balance_classes = False,
                                      refit_strategy = None,
                                      kernel = None,
                                      gamma = 'scale',
                                      bias_mitigation = None,
                                      fairness_param = .0,
                                      ):
        str_name_suffix = ''
        fixed_parameters = {"gurobi_model_params": {'Method': -1, 'Threads': 30},
                            "balanced_class_weights": balance_classes,
                            'kernel': kernel,
                            "verbose": False}
        cv_parameters = {}

        if balance_classes:
            str_name_suffix += '_balancedClasses'

        if refit_strategy is not None:
            assert isinstance(refit_strategy, str)
            str_name_suffix += f"_{refit_strategy.replace('_','')}"
        
        if kernel is not None:
            assert isinstance(kernel, str)
            str_name_suffix += f"_{kernel}"
            if kernel == 'gaussian':
                if gamma == 'cv':
                    cv_parameters['gamma'] = np.logspace(start=-4, stop=3, num=10, endpoint=True)
                    str_name_suffix += "Gammacrossval"
                else:
                    fixed_parameters['gamma'] = gamma
                    str_name_suffix += f"Gamma{gamma}"
                    
        if soft_margin == 'cv':
            cv_parameters['c'] = np.logspace(start=-4, stop=3, num=15, endpoint=True)
            str_name_suffix += f'_Ccrossval'
        else:
            fixed_parameters['c'] = float(soft_margin)
            str_name_suffix += f"_C{fixed_parameters['c']}"

        if bias_mitigation is not None:
            if bias_mitigation.startswith('invertedHinges'):
                if bias_mitigation.endswith('full') or bias_mitigation.endswith('0') or bias_mitigation.endswith('1'):
                    fixed_parameters['bias_mitigation'] = bias_mitigation
                else:
                    cv_parameters['bias_mitigation'] = [f"{bias_mitigation}0", f"{bias_mitigation}1"]
                
                if fairness_param == 'cv':
                    cv_parameters['fairness_eps'] = np.linspace(start=0, stop=2, num=9, endpoint=True)[1:]
                    str_name_suffix += f"_{bias_mitigation}-epscrossval"
                else:
                    fixed_parameters['fairness_eps'] = float(fairness_param)
                    str_name_suffix += f"_{bias_mitigation}-eps{fixed_parameters['fairness_eps']}"
            else:
                fixed_parameters['bias_mitigation'] = bias_mitigation
                fixed_parameters['fairness_eps'] = 0.
                str_name_suffix += f"_{bias_mitigation}-eps{fixed_parameters['fairness_eps']}"

        if len(cv_parameters) == 0:
            cv_parameters = None

        return str_name_suffix, fixed_parameters, cv_parameters
    