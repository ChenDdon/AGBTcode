from fairseq.models.roberta import RobertaModel
import argparse
import sys
import numpy as np
import torch


def load_pretrain_model(model_name_or_path, checkpoint_file, data_name_or_path, bpe='smi'):
    '''Currently only load to cpu()'''

    # load model
    pretrain_model = RobertaModel.from_pretrained(
        model_name_or_path,
        checkpoint_file,
        data_name_or_path,  # dict_dir,
        bpe=bpe,
    )
    pretrain_model.eval()
    return pretrain_model


def extract_hidden(pretrain_model, target_file):

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

        hidden_info = all_layer_hiddens['inner_states'][-1]
        # last_hidden shape [tokens_num, sample_num(default=1), hidden_dim]

        # hidden_features.append(hidden_info.squeeze(1).cpu().detach().numpy())
        hidden_features[i] = hidden_info.squeeze(1).cpu().detach().numpy()

    # hidden_features type: dict, length: samples_num
    return hidden_features


def extract_features_from_hidden(hidden_info):

    samples_num = len(hidden_info)
    hidden_dim = np.shape(hidden_info[0])[-1]
    samples_features = np.zeros([samples_num, hidden_dim])
    for n_sample, hidden in hidden_info.items():
        # hidden shape [tokens, embed_dim]
        samples_features[n_sample, :] = hidden[0, :]

    return samples_features


def main(args):
    pretrain_model = load_pretrain_model(
        args.model_name_or_path, args.checkpoint_file, args.data_name_or_path, args.bpe)

    hidden_info = extract_hidden(pretrain_model, args.target_file)

    print('Generate features from hidden information')
    samples_features = extract_features_from_hidden(hidden_info)
    print(f'Features shape: {np.shape(samples_features)}')
    np.save(args.save_feature_path, samples_features)


def parse_args(args):
    parser = argparse.ArgumentParser(description="Tools kit for downstream jobs")

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
    parser.add_argument('--save_feature_path', default='extract_f1.npy', type=str,
                        help="Saving feature filename(path)")
    args = parser.parse_args()
    return args


def cli_main():
    args = parse_args(sys.argv[1:])
    print(args)
    main(args)


if __name__ == '__main__':
    cli_main()
    print('End!')
