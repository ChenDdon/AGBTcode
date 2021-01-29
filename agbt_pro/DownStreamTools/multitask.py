"""
Introduction:
    Apply a multi task machine learning fro pre-trained BERT model
Author:
    Chend
"""


import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import random
import sys
import argparse
import time
import os
from sklearn import metrics
import sklearn.ensemble as ensemble
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr


# Parameter
def parse_args(args):
    parser = argparse.ArgumentParser(description='Transformer for SMILES copy test')

    parser.add_argument('--train_x_datasets', nargs='+', default=['train_x.smi'], type=str,
                        help='Training datasets x for multitask, ordered')
    parser.add_argument('--train_y_datasets', nargs='+', default=['train_y.npy'], type=str,
                        help='Training datasets y for multitask, ordered')
    parser.add_argument('--valid_x_datasets', nargs='+', default=['valid_x.smi'], type=str,
                        help='Testing datasets x for multitask, ordered')
    parser.add_argument('--valid_y_datasets', nargs='+', default=['valid_y.npy'], type=str,
                        help='Testing datasets y for multitask, ordered')
    parser.add_argument('--test_x_datasets', nargs='+', default=None, type=str,
                        help='Testing datasets x for multitask, ordered')
    parser.add_argument('--test_y_datasets', nargs='+', default=None, type=str,
                        help='Testing datasets y for multitask, ordered')
    parser.add_argument('--add_features', default=False, action='store_true')
    parser.add_argument('--add_train_features', default=['train_path.npy'], nargs='+', type=str,
                        help="Additional train features")
    parser.add_argument('--add_valid_features', default=['valid_path.npy'], nargs='+', type=str,
                        help="Additional validation features")
    parser.add_argument('--add_test_features', default=None, nargs='+', type=str,
                        help='Additional test features')
    parser.add_argument('--features_norm', action='store_true', default=False)
    parser.add_argument('--additional_features_norm', action='store_true', default=False)

    parser.add_argument('--save_mode', action='store_true', default=False)
    parser.add_argument('--save_model_path_pre', default='model', type=str,
                        help='Save best model prefix path')
    parser.add_argument('--save_predict_result', action='store_true', default=False)
    parser.add_argument('--save_predict_result_prefix', default='pre_result', type=str,
                        help='Save predict result path, save as .npy')

    parser.add_argument('--mode', default='multitask_iterative', type=str,
                        choices=['multitask_iterative', 'multitask_sequential',
                                 'test', 'single_task', 'no_train', 'GBRT'],
                        help='Choose the mode form choices')
    parser.add_argument('--train_batch', default=8, type=int,
                        help='Batch size in training.')
    parser.add_argument('--valid_batch', default=16, type=int,
                        help='Batch size in validate')
    parser.add_argument('--test_batch', default=16, type=int,
                        help='Batch size in training.')
    parser.add_argument('--neural_num_list', nargs='+', default=[2048, 2048, 1024, 512], type=int,
                        help='inner neual numbers in DNN')
    parser.add_argument('--n_workers', type=int, default=4,
                        help='Number of workers for dataloader')
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='Learning rate for optimizer.')
    parser.add_argument('--dropout', default=0.0, type=float,
                        help='Dropout ratio in training.')
    parser.add_argument('--random_seed', default=12345, type=int,
                        help='Random seed, make sure the reproducibility')
    parser.add_argument('--max_epoch', default=2, type=int,
                        help='Maximum epoch in training')
    parser.add_argument('--cuda_device', default=None, type=str,
                        help='Index of using cuda device')
    parser.add_argument('--lr_decay', default=0.5, type=float,
                        help='decay ratio of learning rate on schedule')
    parser.add_argument('--weight_decay', default=0.0, type=float,
                        help='Weight decay of optimizer.')
    parser.add_argument('--momentum', default=0.0, type=float)
    parser.add_argument('--lr_step', default=300, type=int)
    parser.add_argument('--log_interval', default=1, type=int,
                        help='log progress every N epoches')
    args = parser.parse_args()
    return args


class DeepNN(nn.Module):
    '''
    Down stream model, based on pretrained model
    '''
    def __init__(self,
                 n_features=1024,
                 n_nural_list=[2048, 2048, 1024, 512],
                 n_out=4,
                 dropout=0.01):
        super().__init__()

        self.n_features = n_features
        n_nural_list.insert(0, n_features)
        assert len(n_nural_list) > 1, "Hidden layers should > 0"

        self.fc = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(n_nural_list[i], n_nural_list[i+1]),
                    nn.ReLU(),
                )
                for i in range(len(n_nural_list) - 1)
            ]
        )

        self.out = nn.Linear(n_nural_list[-1], n_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch_size, feature_length]

        for layer in self.fc:
            x = self.dropout(layer(x))
        # x = [batch_size, n_nural_list[-1]]

        x = self.dropout(self.out(x))
        # x = [batch_size, n_out]
        return x


def GBRT(train_x_sets, train_y_sets, valid_x_sets, valid_y_sets, args):
    # default parameters
    gbrt_params = {
        'n_estimators': 10000,
        'max_depth': 5,
        'min_samples_split': 5,
        'learning_rate': 0.01,
        'loss': 'ls',
        'subsample': 0.4,
        'max_features': 'sqrt',
        'criterion': 'friedman_mse',
        'random_state': args.random_seed,
        'n_iter_no_change': 1000,
    }

    for train_x, train_y, valid_x, valid_y in zip(
        train_x_sets, train_y_sets, valid_x_sets, valid_y_sets
    ):
        gbrt = ensemble.GradientBoostingRegressor(**gbrt_params)
        print(np.shape(train_x), np.shape(train_y))
        print(np.shape(valid_x), np.shape(valid_y))
        gbrt.fit(train_x, train_y)
        pred_y = gbrt.predict(valid_x)

        # metrics
        r2 = metrics.r2_score(valid_y, pred_y)
        mae = metrics.mean_absolute_error(valid_y, pred_y)
        mse = metrics.mean_squared_error(valid_y, pred_y)
        p_r = pearsonr(valid_y, pred_y)[0]
        p_r2 = p_r**2
        print("Metrics:")
        print(f"\tr2: {r2:.3f}, mae: {mae:.3f}, mse: {mse:.3f}", end=', ')
        print(f"pearsonr: {p_r:.3f}, pearsonr2: {p_r2:.3f}")


def train(model, iterator, optimizer, criterion, device, k=0):
    model.train()
    epoch_loss = 0
    clip = 1
    for i, (x, y) in enumerate(iterator):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        output = model(x)[:, k].view(-1, 1)
        loss = criterion(output, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device, k=0):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(iterator):
            x = x.to(device)
            y = y.to(device)
            output = model(x)[:, k].view(-1, 1)
            loss = criterion(output, y)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def test(model, iterator, criterion, device, k=0):
    model.eval()
    y_predict = np.array([], dtype=float)
    y_test = np.array([], dtype=float)
    with torch.no_grad():
        for i, (x, y) in enumerate(iterator):
            x = x.to(device)
            output = model(x)[:, k].view(-1, 1)
            y_test = np.append(y_test, y.data.numpy())
            y_predict = np.append(y_predict, output.cpu().data.numpy())
    return y_test, y_predict


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


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
        # nn.init.xavier_normal_(m.weight.data)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(args):
    # Reproducibility
    SEED = args.random_seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Use device
    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data prepare
    print("Data prepare:")
    worker_num = args.n_workers

    # Load features and labels
    assert len(args.train_x_datasets) == len(args.train_y_datasets), "check train data"
    train_x_sets = [np.load(file, allow_pickle=True) for file in args.train_x_datasets]
    train_y_sets = [np.load(file, allow_pickle=True) for file in args.train_y_datasets]

    assert len(args.valid_x_datasets) == len(args.valid_y_datasets), "check valid data"
    valid_x_sets = [np.load(file, allow_pickle=True) for file in args.valid_x_datasets]
    valid_y_sets = [np.load(file, allow_pickle=True) for file in args.valid_y_datasets]

    if args.test_x_datasets is not None:
        assert len(args.valid_y_datasets) == len(args.train_y_datasets), "check test data"
        test_x_sets = [np.load(file, allow_pickle=True) for file in args.test_x_datasets]
        test_y_sets = [np.load(file, allow_pickle=True) for file in args.test_y_datasets]

    # Normilize feature
    if args.features_norm:
        if args.mode in ['multitask_iterative', 'multitask_sequential']:
            print('Features normalization for all sets')
            X_train = np.concatenate(train_x_sets, axis=0)
            X_valid = np.concatenate(valid_x_sets, axis=0)
            if args.test_x_datasets is not None:
                X_test = np.concatenate(test_x_sets, axis=0)
                X_train, X_valid, X_test = data_norm(X_train, X_valid, X_test)
                line_train, line_valid, line_test = 0, 0, 0
                for i, (trainx, validx, testx) in enumerate(
                    zip(train_x_sets, valid_x_sets, test_x_sets)
                ):
                    train_x_sets[i] = X_train[line_train: line_train + trainx.shape[0]]
                    valid_x_sets[i] = X_valid[line_valid: line_valid + validx.shape[0]]
                    test_x_sets[i] = X_test[line_test: line_test + testx.shape[0]]
                    line_train += trainx.shape[0]
                    line_valid += validx.shape[0]
                    line_test += testx.shape[0]
            else:
                X_train, X_valid = data_norm(X_train, X_valid)
                line_train, line_valid = 0, 0
                for i, (trainx, validx) in enumerate(zip(train_x_sets, valid_x_sets)):
                    train_x_sets[i] = X_train[line_train: line_train + trainx.shape[0]]
                    valid_x_sets[i] = X_valid[line_valid: line_valid + validx.shape[0]]
                    line_train += trainx.shape[0]
                    line_valid += validx.shape[0]
        else:
            print('Features normalization for every single set')
            if args.test_x_datasets is not None:
                for i, (trainx, validx, testx) in enumerate(
                    zip(train_x_sets, valid_x_sets, test_x_sets)
                ):
                    train_x_sets[i], valid_x_sets[i], test_x_sets[i] = data_norm(
                        train_x_sets[i], valid_x_sets[i], test_x_sets[i])
            else:
                for i, (trainx, validx) in enumerate(zip(train_x_sets, valid_x_sets)):
                    train_x_sets[i], valid_x_sets[i] = data_norm(
                        train_x_sets[i], valid_x_sets[i])

    # Additional features
    if args.add_features:
        assert (
            len(args.add_train_features) == len(args.add_train_features)
        ), "Number error for add_train_features"
        assert (
            len(args.add_valid_features) == len(args.add_valid_features)
        ), "Number error for add_valid_features"

        additional_train_sets = [
            np.load(i, allow_pickle=True) for i in args.add_train_features]
        additional_valid_sets = [
            np.load(j, allow_pickle=True) for j in args.add_valid_features]
        for i in range(len(additional_train_sets)):
            print(f"Additional train features {i} shape: {additional_train_sets[i].shape}")
            print(f"Additional valid features {i} shape: {additional_valid_sets[i].shape}")

        if args.add_test_features is not None:
            assert (
                len(args.add_test_features) == len(args.add_test_features)
            ), "Number error for add_valid_features"
            additional_test_sets = [
                np.load(k, allow_pickle=True) for k in args.add_test_features]

        # Normilize additional feature
        if args.additional_features_norm:
            print('Apply the normalization of additional features')
            if args.mode in ['multitask_iterative', 'multitask_sequential']:
                print('Apply the normalization of the features')
                X_train = np.concatenate(additional_train_sets, axis=0)
                X_valid = np.concatenate(additional_valid_sets, axis=0)
                if args.add_test_features is not None:
                    X_test = np.concatenate(additional_test_sets, axis=0)
                    X_train, X_valid, X_test = data_norm(X_train, X_valid, X_test)
                    line_train, line_valid, line_test = 0, 0, 0
                    for i, (trainx, validx, testx) in enumerate(
                        zip(additional_train_sets, additional_valid_sets, additional_test_sets)
                    ):
                        additional_train_sets[i] = X_train[line_train: line_train + trainx.shape[0]]
                        additional_valid_sets[i] = X_valid[line_valid: line_valid + validx.shape[0]]
                        additional_test_sets[i] = X_test[line_test: line_test + testx.shape[0]]
                        line_train += trainx.shape[0]
                        line_valid += validx.shape[0]
                        line_test += testx.shape[0]
                else:
                    X_train, X_valid = data_norm(X_train, X_valid)
                    line_train, line_valid = 0, 0
                    for i, (trainx, validx) in enumerate(zip(additional_train_sets, additional_valid_sets)):
                        additional_train_sets[i] = X_train[line_train: line_train + trainx.shape[0]]
                        additional_valid_sets[i] = X_valid[line_valid: line_valid + validx.shape[0]]
                        line_train += trainx.shape[0]
                        line_valid += validx.shape[0]
            else:
                print('Apply the normalization of the every single features')
                if args.test_x_datasets is not None:
                    for i, (trainx, validx, testx) in enumerate(
                        zip(additional_train_sets, additional_valid_sets, additional_test_sets)
                    ):
                        additional_train_sets[i], additional_valid_sets[i], additional_test_sets[i] = data_norm(
                            additional_train_sets[i], additional_valid_sets[i], additional_test_sets[i])
                else:
                    for i, (trainx, validx) in enumerate(zip(train_x_sets, valid_x_sets)):
                        additional_train_sets[i], additional_valid_sets[i] = data_norm(
                            additional_train_sets[i], additional_valid_sets[i])

        train_x_sets = [np.hstack([train_x_sets[i], additional_train_sets[i]])
                        for i in range(len(args.add_train_features))]
        valid_x_sets = [np.hstack([valid_x_sets[j], additional_valid_sets[j]])
                        for j in range(len(args.add_valid_features))]

        if args.add_test_features is not None:
            for i in range(len(additional_test_sets)):
                print(f"Additional test features {i} shape: {additional_test_sets[i].shape}")
            test_x_sets = [np.hstack([test_x_sets[j], additional_test_sets[j]])
                           for j in range(len(args.add_test_features))]

    print(f"train_x_sets shape: {[np.shape(i) for i in train_x_sets]}")
    print(f"train_y_sets shape: {[np.shape(i) for i in train_y_sets]}")
    print(f"valid_x_sets shape: {[np.shape(i) for i in valid_x_sets]}")
    print(f"valid_y_sets shape: {[np.shape(i) for i in valid_y_sets]}")

    if args.test_x_datasets is not None:
        print(f"test_x_sets shape: {[np.shape(i) for i in test_x_sets]}")
        print(f"test_y_sets shape: {[np.shape(i) for i in test_y_sets]}")

    # Datasets and datalader
    train_datasets = [
        Data.TensorDataset(torch.FloatTensor(x), torch.FloatTensor(y[:, np.newaxis]))
        for x, y in zip(train_x_sets, train_y_sets)]
    valid_datasets = [
        Data.TensorDataset(torch.FloatTensor(x), torch.FloatTensor(y[:, np.newaxis]))
        for x, y in zip(valid_x_sets, valid_y_sets)]
    train_iterators = [
        Data.DataLoader(data_set, batch_size=args.train_batch,
                        shuffle=True, num_workers=worker_num)
        for data_set in train_datasets]
    valid_iterators = [
        Data.DataLoader(data_set, batch_size=args.valid_batch,
                        shuffle=False, num_workers=worker_num)
        for data_set in valid_datasets]

    if args.test_x_datasets is not None:
        # datalader
        test_datasets = [
            Data.TensorDataset(torch.FloatTensor(x), torch.FloatTensor(y[:, np.newaxis]))
            for x, y in zip(test_x_sets, test_y_sets)]
        test_iterators = [
            Data.DataLoader(data_set, batch_size=args.test_batch,
                            shuffle=True, num_workers=worker_num)
            for data_set in test_datasets]

    # Hyper parameters
    LEARNING_RATE = args.learning_rate
    DROPOUT = args.dropout
    N_EPOCHS = args.max_epoch
    MODE = args.mode

    # Train, validate and test
    if MODE == 'GBRT':
        print('-'*20 + ' GBRT ' + '-'*20)
        GBRT(train_x_sets, train_y_sets, valid_x_sets, valid_y_sets, args)

    elif MODE == 'multitask_iterative':
        # Make model
        model = DeepNN(n_features=train_x_sets[0].shape[1], n_out=len(train_x_sets),
                       n_nural_list=args.neural_num_list, dropout=DROPOUT)
        print(model)
        model = nn.DataParallel(model).to(device)
        model.apply(initialize_weights)
        print(f'The model has {count_parameters(model):,} trainable parameters')
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=args.momentum)
        print(optimizer)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step, gamma=args.lr_decay)
        print(f"Learning rate scheduler: {lr_scheduler}")

        print('-'*20 + ' Multitask training ' + '-'*20)
        best_valid_loss_small_set = float('inf')
        for epoch in range(N_EPOCHS):
            t0 = time.time()
            train_losses = []
            valid_losses = []
            for k, v in enumerate(train_y_sets):
                train_loss = train(model, train_iterators[k], optimizer, criterion, device, k)
                valid_loss = evaluate(model, valid_iterators[k], criterion, device, k)
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)
            if valid_losses[-1] < best_valid_loss_small_set:
                best_model = model
                best_valid_loss_small_set = valid_losses[-1]
            lr_scheduler.step()
            if epoch % args.log_interval == 0:
                print(f"Epoch: {epoch + 1:04} | Time: {(time.time() - t0):.1f}s | "
                      f"Train MSEloss: {np.round(train_losses, 3)} | "
                      f"Valid MSEloss: {np.round(valid_losses, 3)} | "
                      f"Current lr: {optimizer.param_groups[0]['lr']}")

        # load last/best model
        for i, iterator in enumerate(valid_iterators):
            # y_true, y_predict = test(best_model, iterator, criterion, device, i)
            y_true, y_predict = test(model, iterator, criterion, device, i)
            metrics_func(y_true, y_predict, i)
            if args.save_predict_result:
                save_name = f"{args.save_predict_result_prefix}_data_{i}_seed_{args.random_seed}"
                np.save(save_name, {"predict": y_predict, "true": y_true})

        if args.save_mode:
            if not os.path.exists(os.path.split(args.save_model_path_pre)[0]):
                os.mkdir(os.path.split(args.save_model_path_pre)[0])
            torch.save(best_model, args.save_model_path_pre + '.pt')

    elif MODE == "multitask_sequential":
        # Make model
        model = DeepNN(n_features=train_x_sets[0].shape[1], n_out=1,
                       n_nural_list=args.neural_num_list, dropout=DROPOUT)
        print(model)
        model = nn.DataParallel(model).to(device)
        model.apply(initialize_weights)
        print(f'The model has {count_parameters(model):,} trainable parameters')
        criterion = nn.MSELoss()
        print('-'*20 + ' multitask training ' + '-'*20)
        best_valid_losses = [float('inf')] * len(train_y_sets)
        best_models = [model] * len(train_y_sets)

        for k, v in enumerate(train_y_sets):
            if k > 0:
                model = best_models[k - 1]
            optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=args.momentum)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=args.lr_step, gamma=args.lr_decay)
            print(optimizer)
            print(f"Learning rate scheduler: {lr_scheduler}")

            for epoch in range(N_EPOCHS):
                t0 = time.time()
                train_loss = train(model, train_iterators[k], optimizer, criterion, device)
                valid_loss = evaluate(model, valid_iterators[k], criterion, device)
                if valid_loss < best_valid_losses[k]:
                    best_valid_losses[k] = valid_loss
                    best_models[k] = model
                lr_scheduler.step()
                if epoch % args.log_interval == 0:
                    print(f"Epoch: {epoch + 1:04} | Data: {k} | Time: {(time.time() - t0):.1f} s | "
                          f"Train MSEloss: {train_loss:.3f} | Valid MSEloss: {valid_loss:.3f} | "
                          f"Current lr: {optimizer.param_groups[0]['lr']}")

            y_true, y_predict = test(best_models[k], valid_iterators[k], criterion, device)
            metrics_func(y_true, y_predict, k)
            if args.save_predict_result:
                save_name = f"{args.save_predict_result_prefix}_data_{k}_seed_{args.random_seed}"
                np.save(save_name, {"predict": y_predict, "true": y_true})

            if args.save_mode:
                if not os.path.exists(os.path.split(args.save_model_path_pre)[0]):
                    os.mkdir(os.path.split(args.save_model_path_pre)[0])
                torch.save(best_models[k], args.save_model_path_pre + f"_data_{k}.pt")

    elif MODE == 'single_task':
        print('-'*20 + ' single_task training ' + '-'*20)

        for k, v in enumerate(train_y_sets):
            # Make model
            model = DeepNN(n_features=train_x_sets[0].shape[1], n_out=1,
                           n_nural_list=args.neural_num_list, dropout=DROPOUT)
            model = nn.DataParallel(model).to(device)
            model.apply(initialize_weights)
            criterion = nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=args.momentum)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=args.lr_step, gamma=args.lr_decay)
            if k == 0:
                print(model)
                print(f'The model has {count_parameters(model):,} trainable parameters')
                print(optimizer)
                print(f"Learning rate scheduler: {lr_scheduler}")

            valid_best_loss = float('inf')
            for epoch in range(N_EPOCHS):
                t0 = time.time()
                train_loss = train(model, train_iterators[k], optimizer, criterion, device)
                valid_loss = evaluate(model, valid_iterators[k], criterion, device)
                if valid_loss < valid_best_loss:
                    valid_best_loss = valid_loss
                    best_model = model
                lr_scheduler.step()
                if epoch % args.log_interval == 0:
                    print(f"Epoch: {epoch + 1:04} | Data: {k} | Time: {(time.time() - t0):.1f} s | "
                          f"Train MSEloss: {train_loss:.3f} | Valid MSEloss: {valid_loss:.3f} | "
                          f"Current lr: {optimizer.param_groups[0]['lr']}")

            y_true, y_predict = test(best_model, valid_iterators[k], criterion, device)
            metrics_func(y_true, y_predict, k)
            if args.save_predict_result:
                save_name = f"{args.save_predict_result_prefix}_data_{k}_seed_{args.random_seed}"
                np.save(save_name, {"predict": y_predict, "true": y_true})

            if args.save_mode:
                if not os.path.exists(os.path.split(args.save_model_path_pre)[0]):
                    os.mkdir(os.path.split(args.save_model_path_pre)[0])
                torch.save(best_model, args.save_model_path_pre + f"_data_{k}.pt")

    elif MODE == 'test':
        print('-'*20 + ' Test ' + '-'*20)

        # load model and test
        model.load_state_dict(torch.load(args.save_model_path_pre, map_location=device))
        for i, test_iterator in enumerate(test_iterators):
            y_true, y_predict = test(model, test_iterator, criterion, device)
            metrics_func(y_true, y_predict, i)
            if args.save_predict_result:
                save_name = f"{args.save_predict_result_prefix}_data_{i}_seed_{args.random_seed}"
                np.save(save_name, {"predict": y_predict, "true": y_true})

    elif MODE == 'no_train':
        print('no_training')


def cli_main():
    args = parse_args(sys.argv[1:])
    print(args)
    main(args)


if __name__ == "__main__":
    cli_main()
    print('##### End! #####')
