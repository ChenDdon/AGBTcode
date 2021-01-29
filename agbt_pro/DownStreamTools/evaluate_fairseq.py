import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import sys
import argparse

from fairseq.models.roberta import RobertaModel
from fairseq.data import Dictionary


def parse_args(args):
    parser = argparse.ArgumentParser(description='BERT features ml')

    parser.add_argument('--checkpoint_dir', default='Smile_result', type=str)
    parser.add_argument('--checkpoint_file', default='checkpoint_best.pt', type=str)
    parser.add_argument('--dict_dir', default='Smile_dataset/', type=str)
    parser.add_argument('--dict_file', default='./Smile_dataset/dict.txt', type=str)
    parser.add_argument('--test_file', default='./Smile_dataset/split_chembl26_test.smi', type=str)
    parser.add_argument('--save_features_path', default='./GBRT/Dataset/IGC50_test_roberta_features.npy', type=str)
    parser.add_argument('--choose_func', default='recostructed_rate', type=str)
    args = parser.parse_args()
    return args


def reconstructed_rate(checkpoint_dir, checkpoint_file, dict_dir, dict_file, test_file):
    # load model
    roberta = RobertaModel.from_pretrained(checkpoint_dir, checkpoint_file, dict_dir, bpe='smi')
    # print(roberta)
    print(roberta.task.source_dictionary.symbols)

    roberta.eval()  # disable dropout
    # roberta.cuda()

    # load_dict
    dictionary = Dictionary.load(dict_file)

    # reconstructed rate
    add_bos, add_eos = True, True
    nsamples, ncorrect = 0, 0
    for i, line in enumerate(open(test_file)):
        sentence = line.strip()
        tokens = dictionary.encode_line(line, add_if_not_exist=False, 
                                        append_eos=add_eos, reverse_order=False)
        print(tokens.size())
        if add_bos:
            tokens = torch.cat((torch.IntTensor([dictionary.bos_index]),
                                tokens)).unsqueeze(0).long()
        print(tokens.size())
        out_layer, _ = roberta.model(tokens, features_only=False)
        pred = out_layer.argmax(-1)

        # translate to Smiles
        translate_pred = dictionary.string(pred, bpe_symbol=None, escape_unk=False)

        nsamples += 1
        if (pred == tokens).all():
            ncorrect += 1
            flag = 1
        else:
            flag = 0
            print(i, flag, sentence)
            print('\t' + translate_pred)
        break
    print('Reconstructed rate: ' + str(ncorrect / float(nsamples)))


def extrate_feature(checkpoint_dir, checkpoint_file, dict_dir, 
                    dict_file, test_file, save_features_path):
    # load model
    roberta = RobertaModel.from_pretrained(checkpoint_dir, checkpoint_file, dict_dir)
    # print(roberta)
    # print(roberta.task.source_dictionary.symbols)

    roberta.eval()  # disable dropout
    # roberta.cuda()

    # load_dict
    dictionary = Dictionary.load(dict_file)

    samples_n = sum([1 for _ in open(test_file)])
    sample_features = torch.zeros(samples_n, 2 * roberta.model.args.encoder_embed_dim)

    # pooler
    avg_pool = nn.AdaptiveAvgPool2d((1, roberta.model.args.encoder_embed_dim))
    max_pool = nn.AdaptiveMaxPool2d((1, roberta.model.args.encoder_embed_dim))

    # reconstructed rate
    add_bos, add_eos = True, True
    for i, line in enumerate(open(test_file)):
        sentence = line.strip()
        tokens = dictionary.encode_line(sentence, add_if_not_exist=False, 
                                        append_eos=add_eos, reverse_order=False)
        if add_bos:
            tokens = torch.cat((torch.IntTensor([dictionary.bos_index]), 
                                tokens)).unsqueeze(0).long()
        features, all_layer_hiddens = roberta.model(tokens, features_only=True, return_all_hiddens=True)
        sample_features[i, :] = torch.cat((avg_pool(all_layer_hiddens['inner_states'][-1].transpose(0, 1)),
                                           max_pool(all_layer_hiddens['inner_states'][-1].transpose(0, 1)),
                                        #    avg_pool(all_layer_hiddens['inner_states'][-2].transpose(0, 1)),
                                        #    max_pool(all_layer_hiddens['inner_states'][-2].transpose(0, 1)),
                                        #    avg_pool(all_layer_hiddens['inner_states'][-4].transpose(0, 1)),
                                           ), dim=2)

        # sample_features[i, :] = features[0, 0, :]
        if i % 100 == 0:
            print(i)
    np.save(save_features_path, sample_features.detach().numpy())


def main():
    # All parameters
    args = parse_args(sys.argv[1:])
    print('Running Parameters:')
    for k, v in args._get_kwargs():
        print(f"\t{k}: {v}")

    # Reproducibility
    SEED = 1235
    np.random.seed(SEED)

    # parameter
    checkpoint_dir = args.checkpoint_dir
    checkpoint_file = args.checkpoint_file
    dict_dir = args.dict_dir
    dict_file = args.dict_file
    test_file = args.test_file

    # run
    if args.choose_func == 'recostructed_rate':
        reconstructed_rate(checkpoint_dir, checkpoint_file, dict_dir,
                           dict_file, test_file)
    else:
        save_features_path = args.save_features_path
        extrate_feature(checkpoint_dir, checkpoint_file, dict_dir,
                        dict_file, test_file, save_features_path)


if __name__ == "__main__":
    main()

    # python /gpfs/wscgpfs01/chendo11/test/evaluate_fairseq.py --checkpoint_dir /gpfs/wscgpfs01/chendo11/test/avg_from_chembl26all_sentence_finetune/hidden_info_from_sentence_finetune/sentence_finetune_default_model_0707/ --checkpoint_file checkpoint_best.pt --dict_dir /gpfs/wscgpfs02/chendo11/workspace/matai/Dataset/finetune_data/LD50/ --dict_file /gpfs/wscgpfs02/chendo11/workspace/matai/Dataset/finetune_data/LD50/dict.txt --test_file "/gpfs/wscgpfs02/chendo11/workspace/matai/Dataset/finetune_data/LD50/LD50_test_canonical.smi" 