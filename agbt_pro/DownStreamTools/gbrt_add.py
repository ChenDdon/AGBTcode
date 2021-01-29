"""
GBRT
"""

# Dependece
import numpy as np
import argparse
import sys

import sklearn.ensemble as ensemble
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from scipy.stats import pearsonr
import pickle


# Parameters
def parse_args(args):
    parser = argparse.ArgumentParser(description="GBRT, by using latent space feature")

    parser.add_argument('--train_x', nargs='+', default=["agl_LC50_train.npy"], type=str)
    parser.add_argument('--train_y', nargs='+', default=["LC50_train_y.npy"], type=str)
    parser.add_argument('--test_x', nargs='+', default=["agl_LC50_test.npy"], type=str)
    parser.add_argument('--test_y', nargs='+', default=["LC50_test_y.npy"], type=str)

    parser.add_argument('--add_features', action='store_true', default=False)
    parser.add_argument('--add_train_x', nargs='+', default=['add_features'], type=str)
    parser.add_argument('--add_test_x', nargs='+', default=['add_extra_featurs'], type=str)
    parser.add_argument('--features_norm', action='store_true', default=False)
    parser.add_argument('--additional_features_norm', action='store_true', default=False)

    parser.add_argument('--save_pred_result', action='store_true', default=False)
    parser.add_argument('--save_model_name', default=None, type=str,
                        help='Model name of GBRT regresion function')
    parser.add_argument('--save_predict_result', default=False, action='store_true')
    parser.add_argument('--save_predict_result_prefix', default='pred_result', type=str,
                        help='Save predict result path, save as .npy')

    parser.add_argument('--random_seed', default=12345, type=int,
                        help='Random seed for reproducibility')
    parser.add_argument('--mode', default='train', type=str,
                        help='GBRT mode choose [train, test, consensus]')

    # hyper parameters
    parser.add_argument('--max_depth', nargs='+', default=[7], type=int,
                        help='Maximum tree depth')
    parser.add_argument('--subsample', nargs='+', default=[0.4], type=float,
                        help='Subsample for fitting individual learners')
    parser.add_argument('--min_samples_split', nargs='+', default=[5], type=int,
                        help='Minimum sample num of each leaf node.')

    parser.add_argument('--n_estimators', default=10, type=int,
                        help='Num of estimator for gbrt')
    parser.add_argument('--learning_rate', default=0.01, type=float,
                        help='Learning rate for gbrt')
    parser.add_argument('--criterion', default='friedman_mse', type=str,
                        help='Loss function for gbrt')
    parser.add_argument('--max_features', default='sqrt', type=str,
                        help='Number of features to be considered')
    parser.add_argument('--loss', default='ls', type=str,
                        help='Loss function to be optimized.')
    parser.add_argument('--n_iter_no_change', default=1000, type=int,
                        help='Early stopping will be used to terminate training')

    parser.add_argument('--feature_selection', action='store_true', default=False)
    parser.add_argument('--fusion_size', default=None, type=int,
                        help='fusion size, default = 512')
    parser.add_argument('--consensus_data', default=None, nargs='+', type=str,
                        help='predict result from different features/methods')

    args = parser.parse_args()
    return args


def metrics_func(true_value, predict_value, data_k=0):
    # metrics
    r2 = metrics.r2_score(true_value, predict_value)
    mae = metrics.mean_absolute_error(true_value, predict_value)
    mse = metrics.mean_squared_error(true_value, predict_value)
    rmse = mse ** 0.5
    pearson_r = pearsonr(true_value, predict_value)[0]
    pearson_r2 = pearson_r ** 2

    # print
    print(f"Metric for data {data_k} - r2: {r2:.3f} mae: {mae:.3f} mse: {mse:.3f} "
          f"rmse: {rmse:.3f} pearsonr: {pearson_r:.3f} pearsonr2: {pearson_r2:.3f}")
    return r2, mae, mse, rmse, pearson_r, pearson_r2


def data_norm(*args):
    assert len(args) > 0, "Datasets' length needs > 0"
    scaler = StandardScaler()
    scaler.fit(np.vstack(args))
    if len(args) == 1:
        norm_args = scaler.transform(args[0])
    else:
        norm_args = [scaler.transform(args[i]) for i in range(len(args))]
    return norm_args


def main(args):
    # Reproducibility
    SEED = args.random_seed
    np.random.seed(SEED)

    # Data prepare
    print("Data prepare:")
    MODE = args.mode
    assert len(args.train_x) == len(args.train_y), "check train data"
    assert len(args.test_x) == len(args.test_y), "check test data"
    train_x_sets = [np.load(file, allow_pickle=True) for file in args.train_x]
    test_x_sets = [np.load(file, allow_pickle=True) for file in args.test_x]

    train_y_sets = [np.load(file, allow_pickle=True) for file in args.train_y]
    test_y_sets = [np.load(file, allow_pickle=True) for file in args.test_y]

    if args.features_norm:
        for i, (trainx, testx) in enumerate(zip(train_x_sets, test_x_sets)):
            train_x_sets[i], test_x_sets[i] = data_norm(trainx, testx)

    if args.add_features:
        add_train_x_sets = [
            np.load(f, allow_pickle=True) for f in args.add_train_x]
        add_test_x_sets = [
            np.load(f, allow_pickle=True) for f in args.add_test_x]

        if args.additional_features_norm:
            for i, (add_trainx, add_testx) in enumerate(
                zip(add_train_x_sets, add_test_x_sets)
            ):
                add_train_x_sets[i], add_test_x_sets[i] = data_norm(add_trainx, add_testx)

        train_x_sets = [np.hstack([train_x_sets[i], add_train_x_sets[i]])
                        for i in range(len(args.add_train_x))]
        test_x_sets = [np.hstack([test_x_sets[j], add_test_x_sets[j]])
                       for j in range(len(args.add_test_x))]

    for i, (trainx, testx) in enumerate(zip(train_x_sets, test_x_sets)):
        print(f"\tTrain x shape: {trainx.shape}")
        print(f"\tTest x shape: {testx.shape}")

    # Training
    save_model_name = args.save_model_name
    if MODE == 'train':
        for i, (x_train, y_train, x_test, y_test) in enumerate(
            zip(train_x_sets, train_y_sets, test_x_sets, test_y_sets)
        ):
            # Hyper parameters
            gbrt_params = {
                'n_estimators': args.n_estimators,
                'max_depth': args.max_depth[i],
                'min_samples_split': args.min_samples_split[i],
                'subsample': args.subsample[i],
                'learning_rate': args.learning_rate,
                'loss': args.loss,
                'max_features': args.max_features,
                'criterion': args.criterion,
                'random_state': args.random_seed,
                'n_iter_no_change': args.n_iter_no_change,
            }
            gbrt = ensemble.GradientBoostingRegressor(**gbrt_params)
            gbrt.fit(x_train, y_train)
            pred_y = gbrt.predict(x_test)

            # save model
            if args.save_model_name is not None:
                with open(save_model_name + str(i), 'wb') as fw:
                    pickle.dump(gbrt, fw)

            metrics_func(y_test, pred_y, i)
            if args.save_predict_result:
                save_name = f"{args.save_predict_result_prefix}_data_{i}_seed_{args.random_seed}"
                np.save(save_name, {"predict": pred_y, "true": y_test})

            if args.feature_selection:
                # from sklearn.inspection import permutation_importance
                # result = permutation_importance(
                #     gbrt, x_test, y_test, n_repeats=1, random_state=SEED, n_jobs=2)
                # if 'feature_importance' not in globals():
                #     feature_importance = result.importances_mean
                # else:
                #     feature_importance = np.vstack([feature_importance, result.importances_mean])
                print('Feature selection.')
                if 'feature_importance' not in locals():
                    feature_importance = gbrt.feature_importances_
                else:
                    feature_importance = np.vstack([feature_importance, gbrt.feature_importances_])

        if args.feature_selection:
            print(f'feature importance shape {np.shape(feature_importance)}')
            if args.fusion_size is not None:
                np.save(f'feature_importance_mean_{args.fusion_size}.npy', feature_importance)
                sorted_idx = np.mean(feature_importance, axis=0).argsort()[-args.fusion_size::]

                # save filtered feature
                dataname = ['fusion_bert_LD50', 'fusion_bert_IGC50',
                            'fusion_bert_LC50', 'fusion_bert_LC50DM']
                for i, (x_train, x_test) in enumerate(zip(train_x_sets, test_x_sets)):
                    np.save(dataname[i] + f'_train_x_{args.fusion_size}', x_train[:, sorted_idx])
                    np.save(dataname[i] + f'_test_x_{args.fusion_size}', x_test[:, sorted_idx])
            else:
                np.save('feature_importance_mean.npy', feature_importance)
                sorted_idx = np.mean(feature_importance, axis=0).argsort()[-512::]

                # save filtered feature
                dataname = ['fusion_bert_LD50', 'fusion_bert_IGC50',
                            'fusion_bert_LC50', 'fusion_bert_LC50DM']
                for i, (x_train, x_test) in enumerate(zip(train_x_sets, test_x_sets)):
                    np.save(dataname[i] + '_train_x', x_train[:, sorted_idx])
                    np.save(dataname[i] + '_test_x', x_test[:, sorted_idx])

    elif MODE == 'test':
        print('-'*20 + ' Test ' + '-'*20)

        for i, (x_test, y_test) in enumerate(zip(test_x_sets, test_y_sets)):
            # load model and test
            with open(save_model_name + str(i), 'rb') as fr:
                gbrt = pickle.load(fr)
            pred_y = gbrt.predict(x_test)
            metrics_func(y_test, pred_y, i)

    elif MODE == 'consensus':
        predicted_sets = [pre.split() for pre in args.consensus_data]

        pred_sets = []
        for i, pred_list in enumerate(predicted_sets):
            pred_consensus = np.mean(
                np.vstack([np.load(d, allow_pickle=True) for d in pred_list]), axis=0)
            y_test = test_y_sets[i]

            # Metric
            r2 = metrics.r2_score(y_test, pred_consensus)
            mae = metrics.mean_absolute_error(y_test, pred_consensus)
            mse = metrics.mean_squared_error(y_test, pred_consensus)
            p_r = pearsonr(y_test, pred_consensus)[0]
            p_r2 = p_r**2
            print(f"Metrics for {i}: ", end=' ')
            print(f"r2: {r2:.3f}, mae: {mae:.3f}, mse: {mse:.3f}", end=', ')
            print(f"pearsonr: {p_r:.3f}, pearsonr2: {p_r2:.3f}", end=', ')
            print(args.test_y[i])

            # pred_sets.append([np.load(d, allow_pickle=True) for d in pred_list])
            # for j, pred_data in enumerate(pred_list):
            #     pred_y = np.load(pred_data, allow_pickle=True)


def cli_main():
    args = parse_args(sys.argv[1:])
    print(args)
    main(args)


if __name__ == "__main__":
    cli_main()
    print('##### End! #####')
