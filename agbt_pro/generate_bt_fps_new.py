"""Summary:
        New interface for extracting smiles features from pretrained model
    
    Author:
        Dong Chen
    Creat:
        08-29-2023
    Last modify:
        08-29-2022
    Dependencies:
        python                    3.9.12
        pytorch                   1.13.1
        fairseq                   0.12.2
        numpy                     1.21.5
"""

import os
import torch
from fairseq.models.roberta import RobertaModel
from fairseq.data.encoders import register_bpe
from fairseq.data import data_utils
from fairseq.data import Dictionary
import re
import numpy as np
import argparse
import sys


@register_bpe('smi')
class SMI2BPE(object):
    def __init__(self, args):
        dict_file = os.path.join(args.model.data, 'dict.txt')
        assert os.path.exists(dict_file), f"dict.txt doesn't exists in {args.model.data}"
        self.vocab_dict = Dictionary.load(dict_file)
        SPACE_NORMALIZER = re.compile(r"\s+")
        self.SMI_SYMBOLS = r"Li|Be|Na|Mg|Al|Si|Cl|Ca|Zn|As|Se|se|Br|Rb|Sr|Ag|Sn|Te|te|Cs|Ba|Bi|[\d]|" + r"[HBCNOFPSKIbcnops#%\)\(\+\-\\\/\.=@\[\]]"

    def encode(self, x):
        return ' '.join(self.tokenize_smiles(x))

    def decode(self, x):
        return x

    def tokenize_smiles(self, line):
        line = re.findall(self.SMI_SYMBOLS, line.strip())
        return line

    def is_beginning_of_word(self, x: str) -> bool:
        return self.decode(x).startswith(' ')


class feature_pooler(object):
    def __init__(self, pooler_type='avg'):
        if pooler_type == 'avg':
            self.pooler = self.avg_pooler
        elif pooler_type == 'bos':
            self.pooler = self.bos_pooler
        
    def avg_pooler(self, feature):
        return np.mean(feature, axis=1)
    
    def bos_pooler(self, feature):
        return feature[:, 0, :]


def extract_smiles_embedding(
    model_name_or_path: str, checkpoint_file: str,
    smi_file: str, save_feature_path: str,
    feature_type: str='avg',
):
    # load model
    model = RobertaModel.from_pretrained(model_name_or_path, checkpoint_file)
    model.bpe = SMI2BPE(model.cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # feature pooler
    fp = feature_pooler(feature_type)

    # get feature
    all_smis = [line.strip() for line in open(smi_file)]
    num_smis = len(all_smis)
    all_smi_embedding = np.zeros([num_smis, model.cfg.model.encoder_embed_dim])
    max_position = model.cfg.model.max_positions
    for i in range(0, num_smis):
        tokens_ids = model.encode(all_smis[i])
        if len(tokens_ids) > max_position:
            tokens_ids = torch.cat(
                (tokens_ids[:max_position - 1], tokens_ids[-1].unsqueeze(0)))

        batch_tokens_ids = data_utils.collate_tokens([torch.LongTensor(tokens_ids)], pad_idx=1).to(model.device)
        last_layer_feature = fp.pooler(model.extract_features(batch_tokens_ids).cpu().detach().numpy())
        all_smi_embedding[i, :] = last_layer_feature

    # save feature
    print(f'Features shape: {np.shape(all_smi_embedding)}')
    np.save(save_feature_path, all_smi_embedding)
    return None


def main(args):
    extract_smiles_embedding(
        model_name_or_path=args.model_name_or_path,
        checkpoint_file=args.checkpoint_file,
        smi_file=args.smi_file,
        save_feature_path=args.save_feature_path,
        feature_type=args.feature_type,
    )
    return None


def parse_args(args):
    parser = argparse.ArgumentParser(description="Tools kit for downstream jobs")

    parser.add_argument('--model_name_or_path', default=None, type=str,
                        help='Pretrained model folder, dict.txt should be in the same file')
    parser.add_argument('--checkpoint_file', default='checkpoint_best.pt', type=str,
                        help='Pretrained model name')
    parser.add_argument('--smi_file', default=None, type=str,
                        help="Target file for feature extraction, default format is .smi")
    parser.add_argument('--feature_type', default='avg', type=str,
                        help="avg for the average of the all symbols embedding. bos for the begin of sequence symbol's embedding")
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
