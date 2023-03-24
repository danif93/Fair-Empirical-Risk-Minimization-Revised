# ----- Standard Imports
import pickle
from itertools import product

# ----- Third Party Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----- Library Imports


# ---------------------
# --- Training Pipeline
# ---------------------

def load_data(data_str, label_str, sensitive_str, 
              merge_sensitive = False,
              num_samples = -1, 
              random_seed = 42,
              ):
    filename = f"{data_str}_labels_{label_str}_sensitive_{sensitive_str}"
    with open(f'data/{filename}.pickle', 'rb') as f:
        X, s, y = pickle.load(f)
    
    # ----- Optionally merge sensitive attribute in the training features
    if merge_sensitive:
        if len(np.unique(s)) != 2:
            # TODO: solve exception below
            raise ValueError("non-binary sensitive attribute need to be one-hot encoded before merging")
        X = np.concatenate((X, s[:,None]), axis=1)

    # ----- Randomly select a samples' subset
    if 0 < num_samples < len(X):
        rng = np.random.default_rng(random_seed)
        rand_idxs = rng.choice(len(X), size=num_samples, replace=False)
        X, s, y = X[rand_idxs], s[rand_idxs], y[rand_idxs]
    
    return X, y, s


# ---------------------------
# --- Data Cleaning & Storing
# ---------------------------

def split_label_sensitive_and_store_data(full_df, labels, sensitives,
                                         dataset_name,
                                         one_hot=True,
                                        ):
    for lbl in labels:
        for sens_col in sensitives:

            not_features = [lbl]

            copy_df = full_df.copy()

            if sens_col is not None:
                for sens_idx, sens_val in enumerate(np.unique(copy_df[sens_col])):
                    sens_mask = copy_df[sens_col] == sens_val
                    copy_df.loc[sens_mask, sens_col] = sens_idx
                copy_df[sens_col] = copy_df[sens_col].astype(int)
                not_features.append(sens_col)
            
            if one_hot:
                copy_df = pd.get_dummies(copy_df)

            feature_columns = [feat_lbl 
                               for feat_lbl in copy_df.columns
                               if feat_lbl not in not_features]
            X = copy_df[feature_columns].to_numpy()
            y = copy_df[lbl].to_numpy()
            if sens_col is not None:
                s = copy_df[sens_col].to_numpy()
                to_store = (X,s,y)
            else:
                to_store = (X,y)
                
            print(f"Label: {lbl}{f' - Sensible: {sens_col}' if sens_col is not None else ''}")
            print(f"Feature matrix shape: {X.shape}")
            print(f"Target vector shape: {y.shape}, values and count: {np.unique(y, return_counts=True)}")
            if sens_col is not None:
                print(f"Sensitive vector shape: {s.shape}, values and count: {np.unique(s, return_counts=True)}")
                print(f"Sensitive/Target distribution: {np.unique(np.array([s,y]).T, return_counts=True, axis=0)}")
            print('')
            
            full_path = f"data/{dataset_name}_labels_{lbl}"
            if sens_col is not None:
                full_path += f"_sensitive_{sens_col}"
            full_path += ".pickle"
            pickle.dump(to_store, open(full_path, 'wb'))
                
                
# -----------------
# --- Visualization
# -----------------

def print_eo_ratio(dataframe, label, sensitive):
    n = len(dataframe)
    lbl = dataframe[label].to_numpy()
    un_lbl = np.unique(lbl)
    sns = dataframe[sensitive].to_numpy()
    un_sns = np.unique(sns)

    msk_cl0 = lbl == un_lbl[0]
    msk_cl1 = ~msk_cl0
    
    msk_gr0 = sns == un_sns[0]
    msk_gr1 = ~msk_gr0
    
    pos_gr0 = ((msk_cl1 & msk_gr0).sum()/n).round(3)
    pos_gr1 = ((msk_cl1 & msk_gr1).sum()/n).round(3)
    neg_gr0 = ((msk_cl0 & msk_gr0).sum()/n).round(3)
    neg_gr1 = ((msk_cl0 & msk_gr1).sum()/n).round(3)
    
    print(f"EO pos gr0: {pos_gr0}; EO pos gr1: {pos_gr1};")
    print(f"EO neg gr0: {neg_gr0}; EO neg gr1: {neg_gr1};")
    

def plot_distributions_sunburst(dataframe, label, sensitive):
    target_vect = dataframe[label]
    sens_vect = dataframe[sensitive]
    
    classes, smpls_classes = np.unique(target_vect, return_counts=True)
    pos_msk, neg_msk = target_vect==classes[1], target_vect==classes[0]
    
    colors_dict = {0:{'main':'#31a354', 0:'#74c476', 1:'#006d2c'},
                   1:{'main':'#de2d26', 0:'#fc9272', 1:'#a50f15'}}
    
    plt.title("Sensitive/class samples' distribution")

    groups, smpls_groups = np.unique(sens_vect, return_counts=True)
    msk_0, msk_1 = sens_vect==groups[0], sens_vect==groups[1]

    n_tot = len(dataframe)
    n_neg, n_pos = smpls_classes[0], smpls_classes[1]
    n_0, n_1 = smpls_groups[0], smpls_groups[1]
    n_neg_0, n_neg_1 = (msk_0 & neg_msk).sum(), (msk_1 & neg_msk).sum()
    n_pos_0, n_pos_1 = (msk_0 & pos_msk).sum(), (msk_1 & pos_msk).sum()
        
    size = 0.4
    plt.pie(smpls_classes, labels=classes, labeldistance=0.2,
            autopct='%1.2f%%', pctdistance=0.6, radius=1-size,
            wedgeprops=dict(width=size, edgecolor='w'),
            colors=[colors_dict[0]['main'], colors_dict[1]['main']])
    
    plt.pie([n_neg_0, n_neg_1, n_pos_1, n_pos_0], labels=[groups[0], groups[1], groups[1], groups[0]],
            autopct='%1.2f%%', pctdistance=0.8, radius=1,
            wedgeprops=dict(width=size, edgecolor='w'),
            colors=[colors_dict[i][j] for i,j in [(0,0),(0,1),(1,1),(1,0)]])
    
    tot_dict =  {f"c{classes[0]}": n_neg, f"c{classes[1]}": n_pos,
                 f"g{groups[0]}": n_0, f"g{groups[1]}": n_1,
                 f"{classes[0]}{groups[0]}": n_neg_0, f"{classes[0]}{groups[1]}": n_neg_1,
                 f"{classes[1]}{groups[0]}": n_pos_0, f"{classes[1]}{groups[1]}": n_pos_1}
    
    print(f"Total samples: {n_tot}")
    cl_distr = '\n\t'.join([f"class {c}: {tot_dict[f'c{c}']}" for c in classes])
    sn_distr = '\n\t'.join([f"group {g}: {tot_dict[f'g{g}']}" for g in groups])
    clsn_distr = '\n\t'.join([f"class {c}, group {g}: {tot_dict[f'{c}{g}']}" 
                              for c,g in product(classes, groups)])
    print(f"Class distribution:\n\t{cl_distr}")
    print(f"Sensitive distribution:\n\t{sn_distr}")
    print(f"Class-sensitive distribution:\n\t{clsn_distr}")