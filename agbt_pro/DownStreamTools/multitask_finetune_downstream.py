import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import argparse
import os
import sys
import time

from fairseq.models.roberta import RobertaModel


def main(args):
    # prepare index data
    if args.load_pretrain:
        pretrain_model = load_pretrain_model(
            args.model_name_or_path, args.checkpoint_file, args.data_name_or_path, args.bpe)

    # load multitask finetuned model
    if args.load_finetune_model:
        model = torch.load(
            args.finetune_model_path, map_location=lambda storage, loc: storage)
        input_seq_indexed = prepare_input_data(pretrain_model, args.target_file)
        sample_features = np.zeros(
            [np.shape(input_seq_indexed)[0], pretrain_model.model.args.encoder_embed_dim])
        for i, seq in enumerate(input_seq_indexed):
            features, _ = model(seq)
            sample_features[i, :] = features[:, 0, :]
        np.save(args.save_feature_path, samples_features)


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


def parse_args(args):
    parser = argparse.ArgumentParser(description="Multitask finetune")

    parser.add_argument('--load_finetune_model', default=False, action='store_true')
    parser.add_argument('--cuda_device', default=None, type=str,
                        help='Index of using cuda device')
    parser.add_argument('--load_pretrain', default=False, action='store_true')
    parser.add_argument('--finetune_model_path', default=None, type=str,
                        help='Finetuned model folder')
    parser.add_argument('--model_name_or_path', default=None, type=str,
                        help='Pretrained model folder')
    parser.add_argument('--checkpoint_file', default='checkpoint_best.pt', type=str,
                        help='Pretrained model name')
    parser.add_argument('--data_name_or_path', default=None, type=str,
                        help="Pre-training dataset folder")
    parser.add_argument('--dict_file', default='dict.txt', type=str,
                        help="Pre-training dict filename(full path)")
    parser.add_argument('--bpe', default='smi', type=str)
    parser.add_argument('--traget_file', default='train_x.smi', type=str,
                        help='Target smiles file')

    args = parser.parse_args()
    return args


def cli_main():
    args = parse_args(sys.argv[1:])
    print(args)
    main(args)


if __name__ == '__main__':
    cli_main()
    print('End!')
