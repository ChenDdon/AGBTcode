import torch

from fairseq.models.roberta import RobertaModel


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


def main():
    '''estimate memory usage'''
    model_name_or_path = "/gpfs/wscgpfs02/chendo11/workspace/chembl27_pubchem_zinc_pretrain/Result/Model_chembl27_pubchem_zinc_embed_512/"
    checkpoint_file = 'checkpoint_best.pt'
    data_name_or_path = "/gpfs/wscgpfs02/chendo11/workspace/chembl27_pubchem_zinc_pretrain/Dataset/pubchem_chembl27_zinc_combine/"

    pretrain_model = load_pretrain_model(
        model_name_or_path, checkpoint_file, data_name_or_path)

    pretrain_model.cuda()
    tokens = pretrain_model.encode('CCCCCCCC').cuda()
    _, all_layer_hiddens = pretrain_model.model(
            tokens.unsqueeze(0), features_only=True, return_all_hiddens=True)

    print(f'predict result: {tokens}')
    print(f'gpu memory usage: {torch.cuda.memory_allocated()}')
    print(f'gpu max memory usage: {torch.cuda.max_memory_allocated()}')


if __name__ == "__main__":
    main()
    print('End!')
