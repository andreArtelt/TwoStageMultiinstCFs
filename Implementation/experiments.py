import pickle
import random
from joblib import Parallel, delayed

import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from warren.cf_dice import DiceExplainer
from warren.groupcf_warren import compute_groupcf as compute_groupcf_warren
from ours.cf_clustering import cluster_instances
from ours.ea_mixedvar_groupcf import compute_mixedvar_groupcf
from kanamori.groupcf_kanamori import GroupCF

from datasets import load_attrition_data, load_lawSchool_dataset, load_creditCardClients_dataset



def run_exp(multiinst_method, dataset, cluster_method, k_folds=5):
    results_accuracies = []
    results_global_cfs = []
    results_local_cfs = []

    f_out_path = f"exp-results/{multiinst_method}_{dataset}_{cluster_method}.pickle"

    print(f"Config: {multiinst_method, dataset, cluster_method}")

    # Load data
    if dataset == "attrition":
        X, y, _, features_desc, features_type, features_range = load_attrition_data()
    elif dataset == "credit":
        X, y, _, features_desc, features_type, features_range = load_creditCardClients_dataset()
    elif dataset == "lawschool":
        X, y, _, features_desc, features_type, features_range = load_lawSchool_dataset()

    feature_idx_whitelist = range(X.shape[1])

    # Downsample some data sets for better performance
    sampling = RandomUnderSampler()
    X, y = sampling.fit_resample(X, y)
    idx = random.sample(range(X.shape[0]), k=min(X.shape[0], 500))
    X, y = X[idx, :], y[idx]

    # Cross validation
    kf = KFold(n_splits=k_folds, shuffle=True)
    for train_index, test_index in kf.split(X):
        try:
            # Split data into training and test set
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Deal with imbalanced data sets
            sampling = RandomOverSampler()
            X_train, y_train = sampling.fit_resample(X_train, y_train)

            # Fit classifier
            xgb = XGBClassifier(objective='binary:logistic', silent=True)
            parameters = {'nthread':[4],
                    'learning_rate': [0.03],
                    'max_depth': [4,5],
                    'min_child_weight': [20],
                    'max_delta_step': [8],
                    'gamma':[0.5,1.5, 2],
                    'subsample': [0.2,1],
                    'reg_alpha': [0.7],
                    'n_estimators': [600]
                    }

            xgb_grid = GridSearchCV(xgb,
                                parameters,
                                cv = 2,
                                scoring = 'roc_auc',
                                n_jobs = 5,
                                verbose=False)

            xgb_grid.fit(X_train,y_train)
            clf = xgb_grid.best_estimator_

            # Evaluate model
            y_train_pred = clf.predict(X_train)
            ypred_prop = clf.predict_proba(X_test)[:, 1]
            ypred = clf.predict(X_test)

            auc = roc_auc_score(y_test, ypred_prop)
            f1 = f1_score(y_test, ypred)
            results_accuracies.append(f1)
            print(auc, f1)
            cnf_matrix = confusion_matrix (y_test, ypred)
            print(cnf_matrix)

            # Consider all negatively classified instances
            y_target = 0
            X_instances = X_test[ypred == y_target,:]
            print(f"Set D: {X_instances.shape}")

            # Compute individual counterfactuals
            dice_expl = DiceExplainer(clf, X_train, y_train)
            X_cfs = []
            X_idx = []
            for i in range(X_instances.shape[0]):
                try:
                    X_cfs.append(dice_expl.compute_counterfactual(X_instances[i, :], 1 - y_target)[0].flatten())
                    X_idx.append(i)
                except:
                    pass

            X_instances = X_instances[X_idx,:]  # Remove instances for which no counterfactual was found!
            X_cfs = np.array(X_cfs)

            X_othersamples = X_train[y_train_pred == 1 - y_target,:]

            # Cluster instances
            clustering = cluster_instances(X_instances, X_cfs, method=cluster_method).labels_
            print(f"Clustering: {clustering}")

            # Compute multi-instance counterfactuals
            def compute_multiinstance_cf(X_inst):
                if multiinst_method == "warren":
                    delta_cf, cf_score = compute_groupcf_warren(clf, X_train, y_train, X_inst, 1 - y_target, X_othersamples)
                    cf_size = len(delta_cf) / X_inst.shape[1]
                    return delta_cf, cf_score, cf_size
                elif multiinst_method == "ours":
                    delta_cf, err_rate = compute_mixedvar_groupcf(X_inst, 1-y_target, clf=clf, features_type=features_type, features_range=features_range,
                                                                features_idx_whitelist=feature_idx_whitelist)
                    cf_size = len(list(filter(lambda i: np.abs(delta_cf[i]) >= 1e-5, range(len(delta_cf))))) / X_inst.shape[1]
                    return delta_cf, 1. - err_rate, cf_size
                elif multiinst_method == "kanamori":
                    expl = GroupCF(clf, features_desc, X_train, y_train, 1-y_target)
                    delta_cf, cf_score = expl.compute_explanation(X_inst)
                    cf_size = len(list(filter(lambda i: np.abs(delta_cf[i]) >= 1e-5, range(len(delta_cf))))) / X_inst.shape[1]
                    return delta_cf, cf_score, cf_size

            cluster_size = len(X_instances) # Global
            _, cf_score, cf_size = compute_multiinstance_cf(X_instances)
            print(f"Global: {cluster_size, cf_score, cf_size}")
            results_global_cfs.append((cluster_size, cf_score, cf_size))

            local_cfs = []
            for l in np.unique(clustering): # Local
                idx = clustering == l

                cluster_size = np.sum(idx)
                delta_cf, cf_score, cf_size = compute_multiinstance_cf(X_instances[idx,:])
                print(f"Local: {cluster_size, cf_score, cf_size}")
                local_cfs.append((cluster_size, cf_score, cf_size))
            results_local_cfs.append(local_cfs)
        except Exception as ex:
            print(ex)

    # Store results
    with open(f_out_path, "wb") as f_out:
        pickle.dump({"results_accuracies": results_accuracies,
                     "results_global_cfs": results_global_cfs,
                     "results_local_cfs": results_local_cfs}, f_out)



if __name__ == "__main__":
    configs = []
    for multiinst_method in ["ours"]:#, "warren", "kanamori"]:
        for dataset in ["credit", "attrition", "lawschool"]:
            for cluster_method in ["dbscan-cf", "dbscan-xorig"]:
                configs.append({"multiinst_method": multiinst_method,
                                "dataset": dataset,
                                "cluster_method": cluster_method})

    Parallel(n_jobs=4)(delayed(run_exp)(**param_config) for param_config in configs)
