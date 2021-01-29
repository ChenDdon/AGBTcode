import numpy as np
import time
import shutil
from multiprocessing import Pool
import os
import sys
import argparse


# global parameter
valid_rate = 0.0001


# split find chunk size
def find_offsets(filename, num_chunks):
    '''
        filename: input smiles file
        num_chunks: number of workers
    '''

    with open(filename, 'r', encoding='utf-8') as f:
        size = os.fstat(f.fileno()).st_size
        chunk_size = size // num_chunks
        offsets = [0 for _ in range(num_chunks + 1)]
        for i in range(1, num_chunks):
            f.seek(chunk_size * i)
            safe_readline(f)
            offsets[i] = f.tell()
        return offsets


def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline().strip()  # delete empty line
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


def select_valid_smiles(input_file, train_file_handle, valid_file_handle, offset=0, end=-1):
    global valid_rate
    n_train, n_valid = 0, 0
    with open(input_file, 'r', encoding='utf-8') as f:
        f.seek(offset)
        line = safe_readline(f)
        while line:
            if end > 0 and f.tell() > end:
                break
            if np.random.rand() <= valid_rate:
                select_valid_line = line.strip()
                valid_file_handle.add_line(select_valid_line + '\n')
                n_valid += 1
            else:
                select_train_line = line.strip()
                train_file_handle.add_line(select_train_line + '\n')
                n_train += 1

            line = f.readline()
    return {'n_valid': n_valid, 'n_train': n_train}


class build_converter(object):
    def __init__(self, out_file):
        self._data_file = open(out_file, 'w')
        self._sizes = []

    def add_line(self, line):
        self._data_file.write(line)

    def merge_file_(self, another_file):
        # Concatenate data
        with open(another_file, 'r') as f:
            shutil.copyfileobj(f, self._data_file)

    def finalize(self):
        self._data_file.close()


def single_select_valid_smiles(input_file, output_train_file, output_valid_file, offset, end):
    train_handle = build_converter(output_train_file)
    valid_handle = build_converter(output_valid_file)

    result = select_valid_smiles(input_file, train_handle, valid_handle, offset, end)

    train_handle.finalize()
    valid_handle.finalize()
    return result


def multipro_select_valid_smiles(input_file, output_train_file, output_valid_file, num_workers=1):

    n_train_valid = [0, 0]

    def count_result(worker_result):
        n_train_valid[0] += worker_result['n_train']
        n_train_valid[1] += worker_result['n_valid']

    offsets = find_offsets(input_file, num_workers)
    pool = None
    if num_workers > 1:
        pool = Pool(processes=num_workers - 1)
        for worker_id in range(1, num_workers):
            output_train_file_id = "{}{}".format(output_train_file, worker_id)
            output_valid_file_id = "{}{}".format(output_valid_file, worker_id)

            pool.apply_async(
                single_select_valid_smiles,
                (input_file, output_train_file_id, output_valid_file_id, offsets[worker_id], offsets[worker_id + 1]),
                callback=count_result,
            )
        pool.close()

    train_handle = build_converter(output_train_file)
    valid_handle = build_converter(output_valid_file)

    count_result(select_valid_smiles(input_file, train_handle, valid_handle, offset=0, end=offsets[1]))

    if num_workers > 1:
        pool.join()
        for worker_id in range(1, num_workers):
            temp_train_path = "{}{}".format(output_train_file, worker_id)
            temp_valid_path = "{}{}".format(output_valid_file, worker_id)
            train_handle.merge_file_(temp_train_path)
            valid_handle.merge_file_(temp_valid_path)
            os.remove(temp_train_path)
            os.remove(temp_valid_path)
    train_handle.finalize()
    valid_handle.finalize()

    print(f'| {n_train_valid[0]} smiles as train set | {n_train_valid[1]} smiles as valid set '
          f'| {sum(n_train_valid)} smiles in total')


def parse_args(args):
    parser = argparse.ArgumentParser(description='Split data into train and valid')

    parser.add_argument('--input_file', default='test.smi', type=str)
    parser.add_argument('--output_train_file', default='save_train.smi', type=str)
    parser.add_argument('--output_valid_file', default='save_valid.smi', type=str)
    parser.add_argument('--num_workers', default=1, type=int)

    args = parser.parse_args()
    return args


def main(args):
    in_file = args.input_file
    out_train_file = args.output_train_file
    out_valid_file = args.output_valid_file
    num_workers = args.num_workers
    t1 = time.time()
    multipro_select_valid_smiles(in_file, out_train_file, out_valid_file, num_workers)
    print(f'| Running time: {time.time() - t1:.3f} s')


def cli_main():
    args = parse_args(sys.argv[1:])
    print(args)
    main(args)


if __name__ == "__main__":
    cli_main()
    print('End!')
