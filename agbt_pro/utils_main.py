"""
Tools kit for downstream jobs
"""


import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import sys
import ripser as rp

from fairseq.models.roberta import RobertaModel
from fairseq import utils


class RobertaMultitaskHead(nn.Module):
    """Head for sentence-level Regression tasks."""

    def __init__(self, input_dim, inner_dim, num_datasets, activation_fn, pooler_dropout):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_datasets)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]   # shape [batch_size, inner_dim]
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


def prepare_input_data(pretrain_model, target_file):

    sample_num = 0
    for i, line in enumerate(open(target_file)):
        if len(line.strip()) == 0:
            continue
        sample_num += 1
    input_seq = np.ones([sample_num, pretrain_model.args.max_positions])

    for i, line in enumerate(open(target_file)):
        if len(line.strip()) == 0:
            continue
        tokens = pretrain_model.encode(line.strip())
        if len(tokens) > pretrain_model.args.max_positions:
            tokens = torch.cat(
                (tokens[:pretrain_model.args.max_positions - 1], tokens[-1].unsqueeze(0)))

        input_seq[i, 0: len(tokens)] = tokens

    return input_seq


def arange_hidden_info(pretrain_model, target_file, hidden_info):
    ''' arange_hidden_info for symbols specific features'''

    sample_num = 0
    for i, line in enumerate(open(target_file)):
        if len(line.strip()) == 0:
            continue
        sample_num += 1
    dict = pretrain_model.task.source_dictionary.symbols
    print(f'Dict: {dict}')
    dict_size = 54  # len(dict)
    arange_features = np.zeros(
        [sample_num, dict_size, pretrain_model.model.args.encoder_embed_dim])

    for i, line in enumerate(open(target_file)):
        if len(line.strip()) == 0:
            continue
        tokens = pretrain_model.encode(line.strip())
        if len(tokens) > pretrain_model.args.max_positions:
            tokens = torch.cat(
                (tokens[:pretrain_model.args.max_positions - 1], tokens[-1].unsqueeze(0)))
        # tokens shape [1, tokens_len]

        for dict_n in range(dict_size):
            location = torch.where(tokens == dict_n)[-1]
            if len(location) == 0:
                continue
            hidden = hidden_info[i]
            # hidden shape [tokens, embed_dim]

            arange_features[i, dict_n, :] = np.mean(hidden[location], axis=0)

    arange_features = np.reshape(arange_features, (sample_num, -1))

    return arange_features


def get_reconstracted_rate(pretrain_model, target_file):

    sample_num = 0
    for i, line in enumerate(open(target_file)):
        if len(line.strip()) == 0:
            continue
        sample_num += 1

    ncorrect = 0
    for i, line in enumerate(open(target_file)):
        if len(line.strip()) == 0:
            continue
        tokens = pretrain_model.encode(line.strip())
        if len(tokens) > pretrain_model.args.max_positions:
            tokens = torch.cat(
                (tokens[:pretrain_model.args.max_positions - 1], tokens[-1].unsqueeze(0)))

        out_layer, _ = pretrain_model.model(tokens.unsqueeze(0), features_only=False)
        pred = out_layer.argmax(-1)

        if (pred == tokens).all():
            ncorrect += 1
        # else:
        #     print(f"{i:4} original | {tokens}")
        #     print(f"{i:4} translate | {pred}")

    print('Reconstructed rate: ' + str(ncorrect / float(sample_num)))

    return None


def extract_hidden(pretrain_model, target_file, args):

    sample_num = 0
    for i, line in enumerate(open(target_file)):
        if len(line.strip()) == 0:
            continue
        sample_num += 1
    hidden_features = {i: None for i in range(sample_num)}

    for i, line in enumerate(open(target_file)):
        if len(line.strip()) == 0:
            continue
        tokens = pretrain_model.encode(line.strip())
        if len(tokens) > pretrain_model.args.max_positions:
            tokens = torch.cat(
                (tokens[:pretrain_model.args.max_positions - 1], tokens[-1].unsqueeze(0)))

        _, all_layer_hiddens = pretrain_model.model(
            tokens.unsqueeze(0), features_only=True, return_all_hiddens=True)

        hidden_info = all_layer_hiddens['inner_states'][args.target_hidden]
        # last_hidden shape [tokens_num, sample_num(default=1), hidden_dim]

        # hidden_features.append(hidden_info.squeeze(1).cpu().detach().numpy())
        hidden_features[i] = hidden_info.squeeze(1).cpu().detach().numpy()

    # hidden_features type: dict, length: samples_num
    return hidden_features


def extract_features_from_hidden(args, hidden_info):

    extract_method = args.extract_features_method

    samples_num = len(hidden_info)
    hidden_dim = np.shape(hidden_info[0])[-1]
    if extract_method == 'agf':
        samples_features = np.zeros(
            [samples_num, hidden_dim * len(args.agf_eigenvalues_statistic)])
        for n_sample, hidden in hidden_info.items():
            # hidden shape [tokens, embed_dim]

            statistic_info = np.zeros(
                [len(args.agf_eigenvalues_statistic), hidden_dim], dtype=float)

            for n_col in range(hidden_dim):
                statistic_eigenvalues = \
                    algebraic_graph_features(np.reshape(hidden[:, n_col], (-1, 1)))
                statistic_info[:, n_col] = \
                    [statistic_eigenvalues[k] for k in args.agf_eigenvalues_statistic]
            samples_features[n_sample, :] = np.reshape(statistic_info, (1, -1))
    elif extract_method == 'average_all':
        samples_features = np.zeros([samples_num, hidden_dim])
        for n_sample, hidden in hidden_info.items():
            # hidden shape [tokens, embed_dim]
            samples_features[n_sample, :] = np.mean(hidden, axis=0)
    elif extract_method == 'bos':
        samples_features = np.zeros([samples_num, hidden_dim])
        for n_sample, hidden in hidden_info.items():
            # hidden shape [tokens, embed_dim]
            samples_features[n_sample, :] = hidden[0, :]

    return samples_features


def algebraic_graph_features(cloudpoints):
    from scipy.spatial import distance
    con_dis = distance.pdist(cloudpoints, 'minkowski', p=1)
    squar_dis = distance.squareform(con_dis, force='tomatrix')
    eigenvalues = np.linalg.eigvals(squar_dis)

    statistic_eigenvalues = {
        'max': np.max(eigenvalues),
        'min': np.min(list(filter(lambda v: v > 0, eigenvalues))),
        'avg': np.mean(eigenvalues),
        'var': np.var(eigenvalues),
        'sum': np.sum(eigenvalues),
    }

    return statistic_eigenvalues


def ph_features(cloudpoints, max_dim=2, dis_metric=None):
    '''persistent homology information'''

    if dis_metric == 'wasserstein':
        from scipy.stats import wasserstein_distance
        ph_info = rp.ripser(cloudpoints, maxdim=max_dim, metric=wasserstein_distance)
    elif dis_metric == 'KL_divergence':
        from scipy.stats import entropy
        from scipy.spatial import distance
        con_dis = distance.pdist(cloudpoints, entropy)
        squar_dis = distance.squareform(con_dis, force='tomatrix')
        ph_info = rp.ripser(squar_dis, maxdim=max_dim, distance_matrix=True)
    elif dis_metric == 'minkowski':
        # in minkowski metric default p=1
        ph_info = rp.ripser(cloudpoints, maxdim=max_dim, metric='minkowski')
    elif dis_metric == 'cosine':
        ph_info = rp.ripser(cloudpoints, maxdim=max_dim, metric='cosine')
    else:
        ph_info = rp.ripser(cloudpoints, maxdim=max_dim, metric='euclidean')
    # ph_info.dict_keys['dgms', 'cocycles', 'num_edges', 'dperm2all', 'idx_perm', 'r_cover']

    def statistical_feature(betti_num):
        if len(betti_num) == 0:
            betti_feature = np.zeros([15, ])  # total features are
        else:
            # birth
            min_birth, max_birth = np.min(betti_num[:, 0]), np.max(betti_num[:, 0])
            mean_birth, var_birth = np.mean(betti_num[:, 0]), np.var(betti_num[:, 0])
            sum_birth = np.sum(betti_num[:, 0])
            # death
            min_death, max_death = np.min(betti_num[:, 1]), np.max(betti_num[:, 1])
            mean_death, var_death = np.mean(betti_num[:, 1]), np.var(betti_num[:, 1])
            sum_birth = np.sum(betti_num[:, 1])
            # ph_length
            ph_length = betti_num[:, 1] - betti_num[:, 0]
            min_length, max_length = np.min(ph_length), np.max(ph_length)
            mean_length, var_length = np.mean(ph_length), np.var(ph_length)
            sum_length = np.sum(ph_length)

            betti_feature = np.array([
                min_birth, max_birth, mean_birth, var_birth, sum_birth,
                min_death, max_death, mean_death, var_death, sum_birth,
                min_length, max_length, mean_length, var_length, sum_length,
            ])
        return betti_feature

    dgms_data = ph_info['dgms']
    if len(dgms_data) == 0:
        features = np.zeros([np.shape(cloudpoints)[0] - 1 + 15*2, ])
        print("dgms shape is 0")
    else:
        betti0_features = np.sort(dgms_data[0][:, 1])[0:-1]
        betti1_features = statistical_feature(dgms_data[1])
        betti2_features = statistical_feature(dgms_data[2])
        features = np.concatenate([betti0_features, betti1_features, betti2_features])
    return features


def load_pretrain_model(model_name_or_path, checkpoint_file, data_name_or_path, bpe='smi'):
    '''Currently only load to cpu()'''

    # load model
    pretrain_model = RobertaModel.from_pretrained(
        model_name_or_path,
        checkpoint_file,
        data_name_or_path,  # dict_dir,
        bpe='smi',
    )
    pretrain_model.eval()
    return pretrain_model


def regression_predict(
    pretrain_model, target_file, save_regression_predict_path,
    regression_label=False, regression_label_path=None
):
    # print(pretrain_model)

    sample_num = 0
    for i, line in enumerate(open(target_file)):
        if len(line.strip()) == 0:
            continue
        sample_num += 1
    predict_value = np.zeros([sample_num, ])

    for i, line in enumerate(open(target_file)):
        if len(line.strip()) == 0:
            continue
        tokens = pretrain_model.encode(line.strip())
        if len(tokens) > pretrain_model.args.max_positions:
            tokens = torch.cat(
                (tokens[:pretrain_model.args.max_positions - 1], tokens[-1].unsqueeze(0)))

        predict_value[i] = pretrain_model.predict(
            'sentence_classification_head', tokens, return_logits=True)
    print(f"predict shape: {np.shape(predict_value)}")
    if regression_label:
        np.save(
            save_regression_predict_path,
            {'predict': predict_value, 'true': np.load(regression_label_path, allow_pickle=True)})
    else:
        np.save(save_regression_predict_path, {'predict': predict_value})


def parse_args(args):
    parser = argparse.ArgumentParser(description="Tools kit for downstream jobs")

    parser.add_argument('--load_pretrain', default=False, action='store_true')
    parser.add_argument('--model_name_or_path', default=None, type=str,
                        help='Pretrained model folder')
    parser.add_argument('--checkpoint_file', default='checkpoint_best.pt', type=str,
                        help='Pretrained model name')
    parser.add_argument('--data_name_or_path', default=None, type=str,
                        help="Pre-training dataset folder")
    parser.add_argument('--dict_file', default='dict.txt', type=str,
                        help="Pre-training dict filename(full path)")
    parser.add_argument('--bpe', default='smi', type=str)
    parser.add_argument('--target_file', default=None, type=str,
                        help="Target file for feature extraction, default format is .smi")
    parser.add_argument('--get_hidden_info', default=False, action='store_true')
    parser.add_argument('--get_hidden_info_from_model', default=False, action='store_true')
    parser.add_argument('--get_hidden_info_from_file', default=False, action='store_true')
    parser.add_argument('--get_reconstracted_rate', default=False, action='store_true')
    parser.add_argument('--save_hidden_info_path', default='hidden_info.npy', type=str)
    parser.add_argument('--hidden_info_path', default='hidden_info.npy', type=str)
    parser.add_argument('--target_hidden', default=-1, type=int,
                        help='Target hidden layer to extract features.')
    parser.add_argument('--extract_features_method', default='average_all', type=str,
                        help='select from [average_all, agf]')
    parser.add_argument('--agf_eigenvalues_statistic', nargs='+', default=['max'], type=str,
                        help='Statistic info of algebraic_graph eigenvalues [max, min, avg, sum, var]')
    parser.add_argument('--extract_features_from_hidden_info', default=False, action='store_true')
    parser.add_argument('--save_feature_path', default='extract_f1.npy', type=str,
                        help="Saving feature filename(path)")
    parser.add_argument('--dis_metric', default=None, type=str,
                        help='wasserstein, KL_divergence, minkowski, cosine, default: euclidean')
    parser.add_argument('--arange_hidden_info_features', default=False, action='store_true')
    parser.add_argument('--save_arange_features_path', default='arange_features.npy', type=str,
                        help='Arange hidden info, for symbol specific features')
    
    parser.add_argument('--regression_predict', default=False, action='store_true')
    parser.add_argument('--regression_label', default=False, action='store_true')
    parser.add_argument('--regression_label_path', default=None, type=str,
                        help='default format .npy')
    parser.add_argument('--save_regression_predict_path', default=None, type=str)
    
    # multitask finetune model
    parser.add_argument('--load_finetune_model', default=False, action='store_true')
    parser.add_argument('--finetune_model_path', default=None, type=str,
                        help='Finetuned model folder')

    args = parser.parse_args()
    return args


def main(args):
    if args.load_pretrain:
        pretrain_model = load_pretrain_model(
            args.model_name_or_path, args.checkpoint_file, args.data_name_or_path, args.bpe)

    if args.get_hidden_info:
        if args.get_hidden_info_from_model:
            hidden_info = extract_hidden(pretrain_model, args.target_file, args)
            np.save(args.save_hidden_info_path, hidden_info)

        if args.get_hidden_info_from_file:
            assert os.path.exists(args.hidden_info_path), "Hidden info not exists"
            print(f'Hidden info loaded from {args.hidden_info_path}')
            hidden_info = np.load(args.hidden_info_path, allow_pickle=True).item()

    if args.extract_features_from_hidden_info:
        print('Generate features from hidden information')
        samples_features = extract_features_from_hidden(args, hidden_info)
        print(f'Features shape: {np.shape(samples_features)}')
        np.save(args.save_feature_path, samples_features)

    if args.load_finetune_model:
        # prepare indexed seq
        input_seq_indexed = prepare_input_data(pretrain_model, args.target_file)

        # load multitask finetuned model
        print('Load multitask finetune model')
        model = torch.load(
            args.finetune_model_path, map_location=lambda storage, loc: storage).module.to('cpu')
        model.eval()
        samples_features = np.zeros(
            [np.shape(input_seq_indexed)[0], pretrain_model.model.args.encoder_embed_dim])
        for i, seq in enumerate(input_seq_indexed):
            features, _ = model(torch.tensor([seq]).long(), features_only=True)
            samples_features[i, :] = features[:, 0, :].detach().numpy()
        np.save(args.save_feature_path, samples_features)

    if args.arange_hidden_info_features:
        print('Arange hidden info for symbols specific features')
        assert args.load_pretrain, "Must load pretrain model"
        assert os.path.exists(args.target_file), "Load target file"
        arange_features = arange_hidden_info(
            pretrain_model, args.target_file, hidden_info)
        np.save(args.save_arange_features_path, arange_features)

    if args.get_reconstracted_rate:
        get_reconstracted_rate(pretrain_model, args.target_file)

    if args.regression_predict:
        regression_predict(
            pretrain_model,
            args.target_file,
            args.save_regression_predict_path,
            args.regression_label,
            args.regression_label_path,
        )


def cli_main():
    args = parse_args(sys.argv[1:])
    print(args)
    main(args)


if __name__ == '__main__':
    cli_main()
    print('End!')

    # example code
    # python "/gpfs/wscgpfs02/chendo11/workspace/matai/fairseq_pro/utils_main.py" --load_pretrain --model_name_or_path /gpfs/wscgpfs01/chendo11/test/bos_finetune_from_chembl26all/finetune_info_from_bos_finetune/bos_finetune_train_4_data_seperate_5000_updates/bos_finetune_IGC50/ --checkpoint_file checkpoint_best.pt --data_name_or_path /gpfs/wscgpfs02/chendo11/workspace/matai/Dataset/finetune_data/IGC50/ --bpe smi --target_file "/gpfs/wscgpfs02/chendo11/workspace/matai/Dataset/finetune_data/IGC50/IGC50_test_canonical.smi" --regression_predict --save_regression_predict_path /gpfs/wscgpfs01/chendo11/test/bos_finetune_from_chembl26all/finetuned_direct_result/IGC50_predict_bos_finetune.npy --regression_label --regression_label_path "/gpfs/wscgpfs02/chendo11/workspace/matai/Dataset/finetune_data/IGC50/IGC50_test_y.npy"
    
