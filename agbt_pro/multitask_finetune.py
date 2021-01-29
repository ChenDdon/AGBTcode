import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import argparse
import os
import sys
import time

from fairseq.models.roberta import RobertaModel
from fairseq import utils


def main(args):
    # Reproducibility
    SEED = args.random_seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Use device
    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    if args.load_pretrain:
        pretrain_model = load_pretrain_model(
            args.model_name_or_path, args.checkpoint_file, args.data_name_or_path, args.bpe)

    # add multitask layer finetune
    pretrain_model.model.classification_heads['multitask_head'] = RobertaMultitaskHead(
        input_dim=pretrain_model.model.args.encoder_embed_dim,
        inner_dim=pretrain_model.model.args.encoder_embed_dim,
        num_datasets=len(args.train_x_datasets),
        activation_fn=pretrain_model.model.args.activation_fn,
        pooler_dropout=0.0,
    )
    print(pretrain_model.model)
    model = nn.DataParallel(pretrain_model.model).to(device)

    # Data prepare
    print("Data prepare:")
    worker_num = args.n_workers

    # Load features and labels
    if args.indexed_data:
        assert len(args.train_x_datasets) == len(args.train_y_datasets), "check train data"
        train_x_sets = [np.load(file, allow_pickle=True) for file in args.train_x_datasets]
        train_y_sets = [np.load(file, allow_pickle=True) for file in args.train_y_datasets]

        assert len(args.valid_x_datasets) == len(args.valid_y_datasets), "check valid data"
        valid_x_sets = [np.load(file, allow_pickle=True) for file in args.valid_x_datasets]
        valid_y_sets = [np.load(file, allow_pickle=True) for file in args.valid_y_datasets]
    elif args.raw_data:
        assert len(args.train_x_datasets) == len(args.train_y_datasets), "check train data"
        train_x_sets = [prepare_input_data(pretrain_model, file)
                        for file in args.train_x_datasets]
        train_y_sets = [np.load(file, allow_pickle=True) for file in args.train_y_datasets]

        assert len(args.valid_x_datasets) == len(args.valid_y_datasets), "check valid data"
        valid_x_sets = [prepare_input_data(pretrain_model, file)
                        for file in args.valid_x_datasets]
        valid_y_sets = [np.load(file, allow_pickle=True) for file in args.valid_y_datasets]
        pass
    else:
        print('Input data format is not right.')

    print(f"train_x_sets shape: {[np.shape(i) for i in train_x_sets]}")
    print(f"train_y_sets shape: {[np.shape(i) for i in train_y_sets]}")
    print(f"valid_x_sets shape: {[np.shape(i) for i in valid_x_sets]}")
    print(f"valid_y_sets shape: {[np.shape(i) for i in valid_y_sets]}")

    # Datasets and datalader
    train_datasets = [
        Data.TensorDataset(torch.FloatTensor(x).long(), torch.FloatTensor(y[:, np.newaxis]))
        for x, y in zip(train_x_sets, train_y_sets)]
    valid_datasets = [
        Data.TensorDataset(torch.FloatTensor(x).long(), torch.FloatTensor(y[:, np.newaxis]))
        for x, y in zip(valid_x_sets, valid_y_sets)]
    train_iterators = [
        Data.DataLoader(data_set, batch_size=args.train_batch,
                        shuffle=True, num_workers=worker_num)
        for data_set in train_datasets]
    valid_iterators = [
        Data.DataLoader(data_set, batch_size=args.valid_batch,
                        shuffle=False, num_workers=worker_num)
        for data_set in valid_datasets]

    # Hyper parameters
    LEARNING_RATE = args.learning_rate
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.0, eps=1e-6, betas=(0.9, 0.99))
    optim_schedule = ScheduledOptim(
        optimizer, model.module.args.encoder_embed_dim, args.n_warmup_steps, factor=1)

    # Training
    best_valid_losses_avg = float('inf')
    for epoch in range(args.num_epoch):
        start_time = time.time()
        train_losses = []
        valid_losses = []
        for k, v in enumerate(train_y_sets):
            train_loss = train(
                model, train_iterators[k], criterion, optim_schedule, device, k)
            valid_loss = evaluate(
                model, valid_iterators[k], criterion, device, k)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
        if np.mean(valid_losses) < best_valid_losses_avg:
            best_model = model
            best_valid_losses_avg = np.mean(valid_losses)

        if epoch % 2 == 0:
            end_time = time.time()
            print(f"Epoch: {epoch + 1:04} | Time: {(end_time - start_time):.1f}s | "
                  f"Train MSEloss: {np.round(train_losses, 3)} | "
                  f"Valid MSEloss: {np.round(valid_losses, 3)} | "
                  f"Current lr: {optimizer.param_groups[0]['lr']}")

    if not os.path.exists(os.path.split(args.save_model_path_pre)[0]):
        os.mkdir(os.path.split(args.save_model_path_pre)[0])
    torch.save(best_model, args.save_model_path_pre + '.pt')
    # torch.load('my_file.pt', map_location=lambda storage, loc: storage)


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


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps, factor=1):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.factor = factor
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self, steps=None):
        if steps is None:
            steps = self.n_current_steps
        return self.init_lr * self.factor * np.min([
            np.power(steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


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


def train(model, iterator, criterion, optim_schedule, device, data_k=0):
    model.train()
    epoch_loss = 0
    for i, (x, y) in enumerate(iterator):
        x = x.to(device)
        y = y.to(device)
        out, _ = model(x, classification_head_name='multitask_head')
        output = out[:, data_k].view(-1, 1)
        loss = criterion(output, y)
        optim_schedule.zero_grad()
        loss.backward()
        optim_schedule.step_and_update_lr()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device, data_k=0):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(iterator):
            x = x.to(device)
            y = y.to(device)
            out, _ = model(x, classification_head_name='multitask_head')
            output = out[:, data_k].view(-1, 1)
            loss = criterion(output, y)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def parse_args(args):
    parser = argparse.ArgumentParser(description="Multitask finetune")

    parser.add_argument('--load_pretrain', default=False, action='store_true')
    parser.add_argument('--cuda_device', default=None, type=str,
                        help='Index of using cuda device')
    parser.add_argument('--model_name_or_path', default=None, type=str,
                        help='Pretrained model folder')
    parser.add_argument('--checkpoint_file', default='checkpoint_best.pt', type=str,
                        help='Pretrained model name')
    parser.add_argument('--data_name_or_path', default=None, type=str,
                        help="Pre-training dataset folder")
    parser.add_argument('--dict_file', default='dict.txt', type=str,
                        help="Pre-training dict filename(full path)")
    parser.add_argument('--bpe', default='smi', type=str)
    parser.add_argument('--raw_data', action='store_true', default=False)
    parser.add_argument('--indexed_data', action='store_true', default=False)
    parser.add_argument('--n_workers', type=int, default=16,
                        help='Number of workers for dataloader')
    parser.add_argument('--train_batch', default=8, type=int,
                        help='Batch size in training.')
    parser.add_argument('--valid_batch', default=16, type=int,
                        help='Batch size in validate')
    parser.add_argument('--train_x_datasets', nargs='+', default=['train_x.smi'], type=str,
                        help='Training datasets x for multitask, ordered')
    parser.add_argument('--train_y_datasets', nargs='+', default=['train_y.npy'], type=str,
                        help='Training datasets y for multitask, ordered')
    parser.add_argument('--valid_x_datasets', nargs='+', default=['valid_x.smi'], type=str,
                        help='Testing datasets x for multitask, ordered')
    parser.add_argument('--valid_y_datasets', nargs='+', default=['valid_y.npy'], type=str,
                        help='Testing datasets y for multitask, ordered')
    parser.add_argument('--random_seed', default=12345, type=int,
                        help='Random seed, make sure the reproducibility')
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='Learning rate for optimizer.')
    parser.add_argument('--n_warmup_steps', default=500, type=int,
                        help="Number of warmup steps in lr scheduler")
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--save_model_path_pre', default='model', type=str)

    args = parser.parse_args()
    return args


def cli_main():
    args = parse_args(sys.argv[1:])
    print(args)
    main(args)


if __name__ == '__main__':
    cli_main()
    print('End!')
    # python -m fairseq_cli.multitask_finetune --load_pretrain --model_name_or_path /gpfs/wscgpfs02/chendo11/workspace/chembl26_training/Result/checkpoints/ --checkpoint_file checkpoint_best_624.pt --data_name_or_path  /gpfs/wscgpfs02/chendo11/workspace/chembl26_training/Dataset/ --raw_data --train_x_datasets /gpfs/wscgpfs02/chendo11/workspace/matai/Dataset/finetune_data/LD50/LD50_train_canonical.smi /gpfs/wscgpfs02/chendo11/workspace/matai/Dataset/finetune_data/IGC50/IGC50_train_canonical.smi /gpfs/wscgpfs02/chendo11/workspace/matai/Dataset/finetune_data/LC50/LC50_train_canonical.smi /gpfs/wscgpfs02/chendo11/workspace/matai/Dataset/finetune_data/LC50DM/LC50DM_train_canonical.smi --train_y_datasets /gpfs/wscgpfs02/chendo11/workspace/matai/Dataset/finetune_data/LD50/LD50_train_y.npy /gpfs/wscgpfs02/chendo11/workspace/matai/Dataset/finetune_data/IGC50/IGC50_train_y.npy /gpfs/wscgpfs02/chendo11/workspace/matai/Dataset/finetune_data/LC50/LC50_train_y.npy /gpfs/wscgpfs02/chendo11/workspace/matai/Dataset/finetune_data/LC50DM/LC50DM_train_y.npy --valid_x_datasets /gpfs/wscgpfs02/chendo11/workspace/matai/Dataset/finetune_data/LD50/LD50_test_canonical.smi /gpfs/wscgpfs02/chendo11/workspace/matai/Dataset/finetune_data/IGC50/IGC50_test_canonical.smi /gpfs/wscgpfs02/chendo11/workspace/matai/Dataset/finetune_data/LC50/LC50_test_canonical.smi /gpfs/wscgpfs02/chendo11/workspace/matai/Dataset/finetune_data/LC50DM/LC50DM_test_canonical.smi --valid_y_datasets /gpfs/wscgpfs02/chendo11/workspace/matai/Dataset/finetune_data/LD50/LD50_test_y.npy /gpfs/wscgpfs02/chendo11/workspace/matai/Dataset/finetune_data/IGC50/IGC50_test_y.npy /gpfs/wscgpfs02/chendo11/workspace/matai/Dataset/finetune_data/LC50/LC50_test_y.npy /gpfs/wscgpfs02/chendo11/workspace/matai/Dataset/finetune_data/LC50DM/LC50DM_test_y.npy --learning_rate 0.00001 --n_warmup_steps 10 --num_epoch 10 --save_model_path_pre /gpfs/wscgpfs02/chendo11/workspace/matai/Result/finetune_checkpoints/toxicity_multitask_finetune_checkpoint/bos/checkpoint_best

    # python -m fairseq_cli.multitask_finetune --load_pretrain --model_name_or_path /gpfs/wscgpfs02/chendo11/workspace/chembl26_training/Result/checkpoints/ --checkpoint_file checkpoint_best_624.pt --data_name_or_path  /gpfs/wscgpfs02/chendo11/workspace/chembl26_training/Dataset/ --raw_data --train_x_datasets /gpfs/wscgpfs02/chendo11/workspace/matai/Dataset/finetune_data/LC50/LC50_train_canonical.smi /gpfs/wscgpfs02/chendo11/workspace/matai/Dataset/finetune_data/LC50DM/LC50DM_train_canonical.smi --train_y_datasets /gpfs/wscgpfs02/chendo11/workspace/matai/Dataset/finetune_data/LC50/LC50_train_y.npy /gpfs/wscgpfs02/chendo11/workspace/matai/Dataset/finetune_data/LC50DM/LC50DM_train_y.npy --valid_x_datasets /gpfs/wscgpfs02/chendo11/workspace/matai/Dataset/finetune_data/LC50/LC50_test_canonical.smi /gpfs/wscgpfs02/chendo11/workspace/matai/Dataset/finetune_data/LC50DM/LC50DM_test_canonical.smi --valid_y_datasets /gpfs/wscgpfs02/chendo11/workspace/matai/Dataset/finetune_data/LC50/LC50_test_y.npy /gpfs/wscgpfs02/chendo11/workspace/matai/Dataset/finetune_data/LC50DM/LC50DM_test_y.npy --learning_rate 0.00001 --n_warmup_steps 10 --num_epoch 4 --save_model_path_pre test_model
