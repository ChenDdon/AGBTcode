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
from scipy.stats import pearsonr

from fairseq.models.roberta import RobertaModel
from fairseq.data import Dictionary


# Parameter
def parse_args(args):
    parser = argparse.ArgumentParser(description='Transformer for SMILES copy test')

    parser.add_argument('--pretrain_model_dirs', nargs='+', default=['../test_data/'], type=str)
    parser.add_argument('--pretrain_model_names', nargs='+', default=['checkpoint_best.pt'], type=str)
    parser.add_argument('--pretrain_data_dir', default='../test_data/', type=str)
    parser.add_argument('--pretrain_dict_path', default='../test_data/dict.txt', type=str)
    parser.add_argument('--train_x_datasets', nargs='+', default=['train_x.smi'], type=str,
                        help='Training datasets x for multitask, ordered')
    parser.add_argument('--train_y_datasets', nargs='+', default=['train_y.npy'], type=str,
                        help='Training datasets y for multitask, ordered')
    parser.add_argument('--valid_x_datasets', nargs='+', default=['valid_x.smi'], type=str,
                        help='Testing datasets x for multitask, ordered')
    parser.add_argument('--valid_y_datasets', nargs='+', default=['valid_y.npy'], type=str,
                        help='Testing datasets y for multitask, ordered')
    parser.add_argument('--test_x_datasets', nargs='+', default=['test_x.smi'], type=str,
                        help='Testing datasets x for multitask, ordered')
    parser.add_argument('--test_y_datasets', nargs='+', default=['test_y.npy'], type=str,
                        help='Testing datasets y for multitask, ordered')
    parser.add_argument('--feature_times', default=2, type=int,
                        help='Muktiples of original features dim')
    parser.add_argument('--feature_mode', default=None, type=str, choices=['multi_avg', 'multi_bos'])
    parser.add_argument('--load_features_from_dir', default=None, type=str,
                        help='Path of features path')
    parser.add_argument('--add_features', default=False, action='store_true')
    parser.add_argument('--add_train_features', default=['train_path.npy'], nargs='+', type=str,
                        help="Additional train features")
    parser.add_argument('--add_valid_features', default=['valid_path.npy'], nargs='+', type=str,
                        help="Additional validation features")
    parser.add_argument('--features_norm', action='store_true', default=False)
    parser.add_argument('--save_model_name', default='./dir/', type=str,
                        help='Save best model')
    parser.add_argument('--save_features_dir', default=None, type=str,
                        help='Save features from pretrained models.')
    parser.add_argument('--mode', default='multitask_iterative', type=str,
                        help='Choose the mode form [multitask_iterative, multitask_sequential]')
    parser.add_argument('--train_batch', default=16, type=int,
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
    parser.add_argument('--dropout', default=0.01, type=float,
                        help='Dropout ratio in training.')
    parser.add_argument('--random_seed', default=12345, type=int,
                        help='Random seed, make sure the reproducibility')
    parser.add_argument('--max_epoch', default=2, type=int,
                        help='Maximum epoch in training')
    parser.add_argument('--cuda_device', default='0', type=str,
                        help='Index of using cuda device')
    parser.add_argument('--lr_milestone', nargs='+', default=[1000], type=int)
    parser.add_argument('--lr_decay', default=0.1, type=float,
                        help='decay ratio of learning rate on schedule')
    args = parser.parse_args()
    return args


def load_pretrain_model(model_name_or_path, checkpoint_file, data_name_or_path):

    # load model
    pretrain_model = RobertaModel.from_pretrained(
        model_name_or_path,
        checkpoint_file,
        data_name_or_path,  # dict_dir,
        bpe='smi',
    )
    pretrain_model.eval()
    return pretrain_model


def extrate_feature(pretrain_model, dict_file, test_file,
                    feature_times=2, feature_mode=None):

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # pretrain_model.to(device)

    # feature continer
    samples_num = sum(1 for _ in open(test_file))
    sample_features = torch.zeros(
        samples_num, feature_times * pretrain_model.model.args.encoder_embed_dim)

    # pooler
    avg_pool = nn.AdaptiveAvgPool2d((1, pretrain_model.model.args.encoder_embed_dim))
    # max_pool = nn.AdaptiveMaxPool2d((1, pretrain_model.model.args.encoder_embed_dim))

    # reconstructed rate
    nsamples, ncorrect = 0, 0

    # extract features
    for i, line in enumerate(open(test_file)):
        tokens = pretrain_model.encode(line.strip())
        if len(tokens) > pretrain_model.args.max_positions:
            tokens = torch.cat(
                (tokens[:pretrain_model.args.max_positions - 1], tokens[-1].unsqueeze(0)))
        features, all_layer_hiddens = pretrain_model.model(
            tokens.unsqueeze(0), features_only=True, return_all_hiddens=True)

        if feature_mode is None:
            sample_features[i, :] = torch.cat([
                avg_pool(all_layer_hiddens['inner_states'][-1].transpose(0, 1)).squeeze(1),
                all_layer_hiddens['inner_states'][-1].transpose(0, 1)[:, 0, :],
            ], dim=1)
        elif feature_mode == 'multi_bos':
            sample_features[i, :] = torch.cat([
                all_layer_hiddens['inner_states'][-j-1].transpose(0, 1)[:, 0, :]
                for j in range(feature_times)
                ], dim=1)
        elif feature_mode == 'multi_avg':
            sample_features[i, :] = torch.cat([
                avg_pool(all_layer_hiddens['inner_states'][-j-1].transpose(0, 1))
                for j in range(feature_times)
                ], dim=2)

        # dictionary = Dictionary.load(dict_file)
        out_layer, _ = pretrain_model.model(tokens.unsqueeze(0), features_only=False)
        pred = out_layer.argmax(-1)
        # translate_pred = dictionary.string(pred, bpe_symbol=None, escape_unk=False)
        nsamples += 1
        if (pred == tokens).all():
            ncorrect += 1
    print(f'Reconstructed rate: {ncorrect / float(nsamples)}')

    # smiles = 'CCccOC'
    # aa = pretrain_model.encode(smiles)
    # bb = pretrain_model.decode(aa)

    return sample_features.cpu().detach().numpy()


class DeepNN(nn.Module):
    '''
    Down stream model, based on pretrained model
    '''
    def __init__(self,
                 n_features=1024,
                 n_nural_list=[2048, 2048, 1024, 512],
                 dropout=0.01):
        super().__init__()

        self.n_features = n_features
        self.fc0 = nn.Linear(n_features, n_nural_list[0])
        assert len(n_nural_list) > 1, "inner layers should > 1"
        self.fc = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(n_nural_list[i], n_nural_list[i+1]),
                    nn.ReLU(),
                )
                for i in range(len(n_nural_list) - 1)
            ]
        )
        # self.fc = nn.ModuleList([nn.Linear(i, i+1)])
        self.out = nn.Linear(n_nural_list[-1], 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch_size, feature_length]

        x = self.dropout(self.fc0(x))
        for layer in self.fc:
            x = self.dropout(layer(x))
        # x = [batch_size, n[-1]]

        x = self.dropout(self.out(x))
        # x = [batch_size, 1]
        return x


def GBRT_test(train_x_sets, train_y_sets, valid_x_sets, valid_y_sets):

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
        'random_state': 12345,
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


def train(model, iterator, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for i, (x, y) in enumerate(iterator):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(iterator):
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            loss = criterion(output, y)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def test(model, iterator, criterion, device):
    model.eval()
    y_predict = np.array([], dtype=float)
    y_test = np.array([], dtype=float)
    with torch.no_grad():
        for i, (x, y) in enumerate(iterator):
            x = x.to(device)
            output = model(x)
            y_test = np.append(y_test, y.data.numpy())
            y_predict = np.append(y_predict, output.cpu().data.numpy())
    # metrics
    r2 = metrics.r2_score(y_test, y_predict)
    mae = metrics.mean_absolute_error(y_test, y_predict)
    mse = metrics.mean_squared_error(y_test, y_predict)
    pearson_r = pearsonr(y_test, y_predict)[0]
    pearson_r2 = pearson_r ** 2
    return r2, mae, mse, pearson_r, pearson_r2


def data_norm(*args):
    assert len(args) > 0, "Dataset shape[0] needs > 0"
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(args[0])
    norm_args = [scaler.transform(args[i]) for i in range(len(args))]
    return norm_args


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
        # nn.init.xavier_normal_(m.weight.data)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def multi_task():
    # Parameters
    args = parse_args(sys.argv[1:])
    print(args)

    # Reproducibility
    SEED = args.random_seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Use device
    cuda_device = args.cuda_device
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data prepare
    print("Data prepare:")
    MODE = args.mode
    worker_num = args.n_workers

    # extrate_feature
    if args.load_features_from_dir is not None:
        assert (
            os.path.exists(os.path.join(args.load_features_from_dir, 'train_xy_sets.npy'))
        ), f"train_xy_sets.npy is not existing in folder {args.load_features_from_dir}"
        assert (
            os.path.exists(os.path.join(args.load_features_from_dir, 'valid_xy_sets.npy'))
        ), f"valid_xy_sets.npy is not existing in folder {args.load_features_from_dir}"
        train_x_sets, train_y_sets = np.load(
            os.path.join(args.load_features_from_dir, 'train_xy_sets.npy'), allow_pickle=True)
        valid_x_sets, valid_y_sets = np.load(
            os.path.join(args.load_features_from_dir, 'valid_xy_sets.npy'), allow_pickle=True)
    else:
        # load pretrained model
        pretrain_models = [
            load_pretrain_model(args.pretrain_model_dirs[i], args.pretrain_model_names[i],
                                args.pretrain_data_dir)
            for i in range(len(args.pretrain_model_dirs))
        ]
        train_x_sets, valid_x_sets = [], []
        if len(args.pretrain_model_dirs) > 1 or len(args.pretrain_model_names) > 1:
            assert len(args.pretrain_model_dirs) == len(args.train_x_datasets), "check pretrian models"
            assert len(args.pretrain_model_dirs) == len(args.valid_x_datasets), "check pretrian models"
            train_x_sets = [
                extrate_feature(pretrain_models[i], args.pretrain_dict_path, args.train_x_datasets[i],
                                feature_times=args.feature_times, feature_mode=args.feature_mode)
                for i in range(len(args.train_x_datasets))
            ]
            valid_x_sets = [
                extrate_feature(pretrain_models[j], args.pretrain_dict_path, args.valid_x_datasets[j],
                                feature_times=args.feature_times, feature_mode=args.feature_mode)
                for j in range(len(args.valid_x_datasets))
            ]
        else:
            print('single model loaded')
            train_x_sets = [
                extrate_feature(pretrain_models[0], args.pretrain_dict_path, args.train_x_datasets[i],
                                feature_times=args.feature_times, feature_mode=args.feature_mode)
                for i in range(len(args.train_x_datasets))
            ]
            valid_x_sets = [
                extrate_feature(pretrain_models[0], args.pretrain_dict_path, args.valid_x_datasets[j],
                                feature_times=args.feature_times, feature_mode=args.feature_mode)
                for j in range(len(args.train_x_datasets))
            ]
        train_y_sets = [np.load(file, allow_pickle=True) for file in args.train_y_datasets]
        valid_y_sets = [np.load(file, allow_pickle=True) for file in args.valid_y_datasets]

    if args.save_features_dir is not None:
        assert os.path.exists(args.save_features_dir), 'save_features_dir is not exists'
        np.save(os.path.join(args.save_features_dir, 'train_xy_sets.npy'),
                [train_x_sets, train_y_sets])
        np.save(os.path.join(args.save_features_dir, 'valid_xy_sets.npy'),
                [valid_x_sets, valid_y_sets])

    # additional features
    if args.add_features:
        assert (
            len(args.add_train_features) == len(args.add_valid_features)
        ), "Number error for add_features"
        assert (
            len(args.add_train_features) == len(train_x_sets)
        ), "Number error for add_features"

        additional_train_sets = [
            np.load(i, allow_pickle=True) for i in args.add_train_features
        ]
        additional_valid_sets = [
            np.load(j, allow_pickle=True) for j in args.add_valid_features
        ]
        for i in range(len(additional_train_sets)):
            print(f"Additional train features {i} shape: {additional_train_sets[i].shape}")
            print(f"Additional valid features {i} shape: {additional_valid_sets[i].shape}")

        train_x_sets = [
            np.hstack([train_x_sets[i], additional_train_sets[i]])
            for i in range(len(train_x_sets))
        ]
        valid_x_sets = [
            np.hstack([valid_x_sets[j], additional_valid_sets[j]])
            for j in range(len(valid_x_sets))
        ]

    # normilize feature
    if args.features_norm:
        for i, (train_x, valid_x) in enumerate(zip(train_x_sets, valid_x_sets)):
            train_x_sets[i], valid_x_sets[i] = data_norm(train_x, valid_x)

    train_data_idx = {k: len(v) for k, v in enumerate(train_y_sets)}
    print(f"train_x_sets shape: {[np.shape(i) for i in train_x_sets]}")
    print(f"train_y_sets shape: {[np.shape(i) for i in train_y_sets]}")
    print(f"valid_x_sets shape: {[np.shape(i) for i in valid_x_sets]}")
    print(f"valid_y_sets shape: {[np.shape(i) for i in valid_y_sets]}")

    # dataset
    train_datasets = []
    for x, y in zip(train_x_sets, train_y_sets):
        # x, y = data_noem(x, y)
        train_datasets.append(
            Data.TensorDataset(torch.FloatTensor(x), torch.FloatTensor(y[:, np.newaxis])))
    valid_datasets = []
    for x, y in zip(valid_x_sets, valid_y_sets):
        # x, y = data_noem(x, y)
        valid_datasets.append(
            Data.TensorDataset(torch.FloatTensor(x), torch.FloatTensor(y[:, np.newaxis])))
    train_iterators = [
        Data.DataLoader(data_set, batch_size=args.train_batch, shuffle=True, num_workers=worker_num)
        for data_set in train_datasets]
    valid_iterators = [
        Data.DataLoader(data_set, batch_size=args.valid_batch, shuffle=False, num_workers=worker_num)
        for data_set in valid_datasets]

    # Hyper parameters
    LEARNING_RATE = args.learning_rate
    DROPOUT = args.dropout
    N_EPOCHS = args.max_epoch

    # Make model
    model = DeepNN(n_features=train_x_sets[0].shape[1], n_nural_list=args.neural_num_list, dropout=DROPOUT)
    print(model)
    model = nn.DataParallel(model).to(device)
    model.apply(initialize_weights)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # Train, validate and test
    save_model_name = args.save_model_name
    if MODE == 'GBRT_test':
        print('-'*20 + ' GBRT_test ' + '-'*20)
        GBRT_test(train_x_sets, train_y_sets, valid_x_sets, valid_y_sets)

    elif MODE == 'multitask_iterative':
        print('-'*20 + ' multitask training ' + '-'*20)
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.5)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_milestone, gamma=args.lr_decay, last_epoch=-1)

        best_valid_losses = [float('inf')] * len(train_y_sets)
        best_valid_losses_sum = float('inf')
        out_layer = [model.module.out] * len(train_y_sets)

        for epoch in range(N_EPOCHS):
            start_time = time.time()
            train_losses = []
            valid_losses = []
            for k, v in train_data_idx.items():
                model.module.out = out_layer[k]
                train_iterator = train_iterators[k]
                valid_iterator = valid_iterators[k]
                train_loss = train(model, train_iterator, optimizer, criterion, device)
                valid_loss = evaluate(model, valid_iterator, criterion, device)
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)
                if valid_loss < best_valid_losses[k]:
                    best_valid_losses[k] = valid_loss
                    out_layer[k] = model.module.out
            if np.sum(valid_losses) < best_valid_losses_sum:
                best_model = model
            lr_scheduler.step()
            if epoch % 10 == 0:
                end_time = time.time()
                print(f"Epoch: {epoch+1:04} | Time: {(end_time-start_time):.1f}s", end=' | ')
                print(f"Train MSEloss: {np.round(train_losses, 3)}", end=" | ")
                print(f"Valid MSEloss: {np.round(valid_losses, 3)} Current lr: {lr_scheduler.get_lr()}")

        # load best model
        test_metrics = []
        for i, layer in enumerate(out_layer):
            best_model.out = layer
            test_iterator = valid_iterators[i]
            test_result = np.round(test(best_model, test_iterator, criterion, device), 3)
            print(f"Metrix {i} : ###", end='\t')
            print(f"r2: {test_result[0]}, mae: {test_result[1]}, mse: {test_result[2]}", end=', ')
            print(f"pearsonr: {test_result[3]}, pearsonr2: {test_result[4]}")
            torch.save(best_model, os.path.join(args.save_model_name, f"iter_model_{i}.pt"))

    elif MODE == 'multitask_sequencial':
        print('-'*20 + ' multitask training ' + '-'*20)
        best_valid_losses = [float('inf')] * len(train_y_sets)
        best_models = [model] * len(train_y_sets)
        out_layer = [model.module.out] * len(train_y_sets)

        for k, v in train_data_idx.items():
            start_time = time.time()
            model.module.out = out_layer[k]
            optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.5)
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=args.lr_milestone, gamma=args.lr_decay, last_epoch=-1)
            train_iterator = train_iterators[k]
            valid_iterator = valid_iterators[k]
            for epoch in range(N_EPOCHS):
                train_loss = train(model, train_iterator, optimizer, criterion, device)
                valid_loss = evaluate(model, valid_iterator, criterion, device)
                if valid_loss < best_valid_losses[k]:
                    best_valid_losses[k] = valid_loss
                    best_models[k] = model
                    out_layer[k] = model.module.out
                lr_scheduler.step()
                if epoch % 20 == 0:
                    end_time = time.time()
                    print(f"Epoch: {epoch + 1:04} | Time: {(end_time - start_time):.1f}s", end=' | ')
                    print(f"Train MSEloss - dataset {k + 1:02}: {np.round(train_loss, 3)}", end=" | ")
                    print(f"Valid MSEloss - dataset {k + 1:02}: {np.round(valid_loss, 3)}"
                          f" Current lr: {lr_scheduler.get_lr()}")

            test_iterator = valid_iterator
            test_result = np.round(test(best_models[k], test_iterator, criterion, device), 3)
            print("Metrix:###", end='\t')
            print(f"r2: {test_result[0]}, mae: {test_result[1]}, mse: {test_result[2]}", end=', ')
            print(f"pearsonr: {test_result[3]}, pearsonr2: {test_result[4]}")

            # save model
            torch.save(best_models[k], os.path.join(args.save_model_name, f"model{k}.pt"))

    elif MODE == 'test':
        print('-'*20 + ' Test ' + '-'*20)
        # load model and test
        model.load_state_dict(torch.load(save_model_name, map_location=device))
        test_metrics = test(model, test_iterator, criterion, device)
        r2, mae, mse, pearson_r, pearson_r2 = test_metrics
        print(f"\tr2: {r2:.3f}, mae: {mae:.3f}, mse: {mse:.3f}", end=', ')
        print(f"pearsonr: {pearson_r:.3f}, pearsonr2: {pearson_r2:.3f}")

    elif MODE == 'no_training':
        print('no_training')


if __name__ == "__main__":
    multi_task()
    print('##### End! #####')
