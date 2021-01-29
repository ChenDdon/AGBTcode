"""
feature_analysis tools:
    - feature selection
"""

import numpy as np
import argparse
import time
import os
import sys
import sklearn.ensemble as ensemble
from sklearn.preprocessing import StandardScaler


def main(args):
    # Reproducibility
    SEED = args.random_seed
    np.random.seed(SEED)

    # Data prepare
    assert len(args.train_x_f1) == len(args.train_y), "check train_x 1"
    assert len(args.train_x_f2) == len(args.train_y), "check train_x 2"
    assert len(args.train_x_f3) == len(args.train_y), "check train_x 3"
    train_x_f1 = [np.load(f, allow_pickle=True) for f in args.train_x_f1]
    train_x_f2 = [np.load(f, allow_pickle=True) for f in args.train_x_f2]
    train_x_f3 = [np.load(f, allow_pickle=True) for f in args.train_x_f3]
    train_y = [np.load(f, allow_pickle=True) for f in args.train_y]

    for i, (x1, x2, x3) in enumerate(zip(train_x_f1, train_x_f2, train_x_f3)):
        print(f"x1 shape: {np.shape(x1)}")
        print(f"x2 shape: {np.shape(x2)}")
        print(f"x3 shape: {np.shape(x3)}")

    assert len(args.test_x_f1) == len(args.test_y), "check test_x 1"
    assert len(args.test_x_f2) == len(args.test_y), "check test_x 2"
    assert len(args.test_x_f3) == len(args.test_y), "check test_x 3"
    test_x_f1 = [np.load(f, allow_pickle=True) for f in args.test_x_f1]
    test_x_f2 = [np.load(f, allow_pickle=True) for f in args.test_x_f2]
    test_x_f3 = [np.load(f, allow_pickle=True) for f in args.test_x_f3]
    # test_y = [np.load(f, allow_pickle=True) for f in args.test_y]

    if args.features_norm:
        def norm_func(train_x_sets, valid_x_sets):
            X_train = np.concatenate(train_x_sets, axis=0)
            X_valid = np.concatenate(valid_x_sets, axis=0)
            X_train, X_valid = data_norm(X_train, X_valid)
            line_train, line_valid = 0, 0
            for i, (trainx, validx) in enumerate(zip(train_x_sets, valid_x_sets)):
                train_x_sets[i] = X_train[line_train: line_train + trainx.shape[0]]
                valid_x_sets[i] = X_valid[line_valid: line_valid + validx.shape[0]]
                line_train += trainx.shape[0]
                line_valid += validx.shape[0]
            return train_x_sets, valid_x_sets

        train_x_f1, test_x_f1 = norm_func(train_x_f1, test_x_f1)
        train_x_f2, test_x_f2 = norm_func(train_x_f2, test_x_f2)
        train_x_f3, test_x_f3 = norm_func(train_x_f3, test_x_f3)

    # combine all x
    train_x_sets = [np.hstack([train_x_f1[i], train_x_f2[i], train_x_f3[i]])
                    for i in range(len(args.train_y))]
    valid_x_sets = [np.hstack([test_x_f1[j], test_x_f2[j], test_x_f3[j]])
                    for j in range(len(args.test_y))]

    # select features
    importances = feature_selection_rf(train_x_sets, train_y, args)
    np.save(
        os.path.join(args.save_folder_path, 'feature_importances_all.npy'),
        importances)
    feature_importances = np.mean(importances, axis=0)
    indices = np.argsort(feature_importances)[::-1]
    selected_indices = indices[0: args.n_select_features]

    # select new features and save new features
    for i, (x1, x2) in enumerate(zip(train_x_sets, valid_x_sets)):
        train_x_new = x1[:, selected_indices]
        test_x_new = x2[:, selected_indices]

        # save new features
        np.save(
            os.path.join(args.save_folder_path, f'fusion_train_x_{i}.npy'),
            train_x_new)
        np.save(
            os.path.join(args.save_folder_path, f'fusion_test_x_{i}.npy'),
            test_x_new)


def feature_selection_rf(x_datasets, y_datasets, args):
    all_feature_importance = []
    for i, (x, y) in enumerate(zip(x_datasets, y_datasets)):
        t0 = time.time()
        rf_params = {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth[i],
            'min_samples_split': args.min_samples_split[i],
            'max_features': 'auto',
            'random_state': args.random_seed,
            'n_jobs': args.n_workers,
        }
        rf = ensemble.ExtraTreesRegressor(**rf_params)
        rf.fit(x, y)
        all_feature_importance.append(rf.feature_importances_)
        print(f"Num {i} data shape: {np.shape(x)} running time: {time.time()-t0}")
    importances = np.vstack(all_feature_importance)
    print(f"Shape of all data importance: {np.shape(importances)}")
    return importances


def data_norm(*args):
    assert len(args) > 0, "Datasets' length needs > 0"
    scaler = StandardScaler()
    scaler.fit(np.vstack(args))
    if len(args) == 1:
        norm_args = scaler.transform(args[0])
    else:
        norm_args = [scaler.transform(args[i]) for i in range(len(args))]
    return norm_args


def parse_args(args):
    parser = argparse.ArgumentParser(description="Feature analysis")

    parser.add_argument('--train_x_f1', nargs='+', default=["LC50_train.npy"], type=str)
    parser.add_argument('--train_x_f2', nargs='+', default=["LC50_train.npy"], type=str)
    parser.add_argument('--train_x_f3', nargs='+', default=["LC50_train.npy"], type=str)
    parser.add_argument('--test_x_f1', nargs='+', default=["LC50_test.npy"], type=str)
    parser.add_argument('--test_x_f2', nargs='+', default=["LC50_test.npy"], type=str)
    parser.add_argument('--test_x_f3', nargs='+', default=["LC50_test.npy"], type=str)
    parser.add_argument('--train_y', nargs='+', default=["LC50_train_y.npy"], type=str)
    parser.add_argument('--test_y', nargs='+', default=["LC50_test_y.npy"], type=str)

    parser.add_argument('--features_norm', action='store_true', default=False)
    parser.add_argument('--save_folder_path', default='./', type=str,
                        help="prefix save new selected features folder")

    parser.add_argument('--n_estimators', default=100, type=int)
    parser.add_argument('--n_workers', default=1, type=int)
    parser.add_argument('--max_depth', nargs='+', default=[7], type=int,
                        help='Maximum tree depth')
    parser.add_argument('--min_samples_split', nargs='+', default=[5], type=int,
                        help='Minimum sample num of each leaf node.')
    parser.add_argument('--random_seed', default=1234, type=int)

    parser.add_argument('--n_select_features', default=512, type=int)

    args = parser.parse_args()
    return args


def cli_main():
    args = parse_args(sys.argv[1:])
    print(args)
    main(args)


if __name__ == "__main__":
    cli_main()
    print('End!')
