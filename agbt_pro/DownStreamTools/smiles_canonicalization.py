import numpy as np
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit import Chem
from rdkit.Chem import Descriptors
import time
import shutil
from multiprocessing import Pool
import os
import sys
import argparse


REMOVER = SaltRemover()
ORGANIC_ATOM_SET = set([1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53])


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
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


# filter smiles functions #####################################################################
def canonical_smile(sml):
    """Helper Function that returns the RDKit canonical SMILES for a input SMILES sequnce.
    Args:
        sml: SMILES sequence.
    Returns:
        canonical SMILES sequnce."""
    return Chem.MolToSmiles(Chem.MolFromSmiles(sml), canonical=True)


def keep_largest_fragment(sml):
    """Function that returns the SMILES sequence of the largest fragment for a input
    SMILES sequnce.
    Args:
        sml: SMILES sequence.
    Returns:
        canonical SMILES sequnce of the largest fragment.
    """
    mol_frags = Chem.GetMolFrags(Chem.MolFromSmiles(sml), asMols=True)
    largest_mol = None
    largest_mol_size = 0
    for mol in mol_frags:
        size = mol.GetNumAtoms()
        if size > largest_mol_size:
            largest_mol = mol
            largest_mol_size = size
    return Chem.MolToSmiles(largest_mol)


def remove_salt_stereo(sml, remover):
    """Function that strips salts and removes stereochemistry information from a SMILES.
    Args:
        sml: SMILES sequence.
        remover: RDKit's SaltRemover object.
    Returns:
        canonical SMILES sequnce without salts and stereochemistry information.
    """
    try:
        sml = Chem.MolToSmiles(remover.StripMol(Chem.MolFromSmiles(sml),
                                                dontRemoveEverything=True),
                               isomericSmiles=False)
        if "." in sml:
            sml = keep_largest_fragment(sml)
    except:
        sml = np.float("nan")
    return sml


def organic_filter(sml):
    """Function that filters for organic molecules.
    Args:
        sml: SMILES sequence.
    Returns:
        True if sml can be interpreted by RDKit and is organic.
        False if sml cannot interpreted by RDKIT or is inorganic.
    """
    try:
        m = Chem.MolFromSmiles(sml)
        atom_num_list = [atom.GetAtomicNum() for atom in m.GetAtoms()]
        is_organic = (set(atom_num_list) <= ORGANIC_ATOM_SET)
        if is_organic:
            return True
        else:
            return False
    except:
        return False


def filter_smiles(sml):
    try:
        m = Chem.MolFromSmiles(sml)
        logp = Descriptors.MolLogP(m)
        mol_weight = Descriptors.MolWt(m)
        num_heavy_atoms = Descriptors.HeavyAtomCount(m)

        # organic smiles
        atom_num_list = [atom.GetAtomicNum() for atom in m.GetAtoms()]
        is_organic = set(atom_num_list) <= ORGANIC_ATOM_SET
        # if (
        #     (logp > -5) & (logp < 7) & (mol_weight > 12) & (mol_weight < 2400) &
        #     (num_heavy_atoms >= 2) & (num_heavy_atoms < 200) & is_organic
        # ):
        if (
            (logp > -5) and (logp < 7) and (mol_weight > 12) and (num_heavy_atoms >=2)
        ):
            return Chem.MolToSmiles(m)
        else:
            return float('nan')
    except:
        return float('nan')


def preprocess_smiles(sml):
    """Function that preprocesses a SMILES string such that it is in the same format as
    the translation model was trained on. It removes salts and stereochemistry from the
    SMILES sequnce. If the sequnce correspond to an inorganic molecule or cannot be
    interpreted by RDKit nan is returned.

    Args:
        sml: SMILES sequence.
    Returns:
        preprocessd SMILES sequnces or nan.
    """
    new_sml = remove_salt_stereo(sml, REMOVER)
    new_sml = filter_smiles(new_sml)
    try:
        canonical_sml = canonical_smile(new_sml)
        return canonical_sml
    except:
        return None

##############################################################################


def run_filter_smiles(input_file, output_file_handle, offset=0, end=-1):

    n_convert, n_failed = 0, 0
    with open(input_file, 'r', encoding='utf-8') as f:
        f.seek(offset)
        line = safe_readline(f)
        while line:
            if end > 0 and f.tell() > end:
                break
            try:
                canonical_line = preprocess_smiles(line.strip().split()[-1])
                if not None:
                    output_file_handle.add_line(canonical_line + '\n')
                    n_convert += 1
                else:
                    n_failed += 1
            except:
                n_failed += 1

            line = f.readline()
    return {'n_convert': n_convert, 'n_failed': n_failed}


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


def single_run_filter_smiles(input_file, output_file, offset, end):
    handle = build_converter(output_file)

    result = run_filter_smiles(input_file, handle, offset, end)

    handle.finalize()
    return result


def multipro_run_filter_smiles(input_file, output_file, num_workers=1):

    n_convert_failed = [0, 0]

    def count_result(worker_result):
        n_convert_failed[0] += worker_result['n_convert']
        n_convert_failed[1] += worker_result['n_failed']

    offsets = find_offsets(input_file, num_workers)
    pool = None
    if num_workers > 1:
        pool = Pool(processes=num_workers - 1)
        for worker_id in range(1, num_workers):
            output_file_id = "{}{}".format(output_file, worker_id)
            pool.apply_async(
                single_run_filter_smiles,
                (input_file, output_file_id, offsets[worker_id], offsets[worker_id + 1]),
                callback=count_result,
            )
        pool.close()

    handle = build_converter(output_file)
    count_result(run_filter_smiles(input_file, handle, offset=0, end=offsets[1]))

    if num_workers > 1:
        pool.join()
        for worker_id in range(1, num_workers):
            temp_file_path = "{}{}".format(output_file, worker_id)
            handle.merge_file_(temp_file_path)
            os.remove(temp_file_path)
    handle.finalize()

    print(f'| {n_convert_failed[0]} smiles converted | {n_convert_failed[1]} smiles are invalid '
          f'| {sum(n_convert_failed)} smiles in total')


def parse_args(args):
    parser = argparse.ArgumentParser(description='Convert SMILES to canonical type')

    parser.add_argument('--input_file', default='test.smi', type=str)
    parser.add_argument('--output_file', default='save_test.smi', type=str)
    parser.add_argument('--num_workers', default=1, type=int)

    args = parser.parse_args()
    return args


def main(args):
    in_file = args.input_file
    out_file = args.output_file
    num_workers = args.num_workers
    t1 = time.time()
    multipro_run_filter_smiles(in_file, out_file, num_workers)
    print(f'| Running time: {time.time() - t1:.3f} s')


def cli_main():
    args = parse_args(sys.argv[1:])
    print(args)
    main(args)


if __name__ == "__main__":

    cli_main()
    print('End!')
    # filename = "/gpfs/wscgpfs02/chendo11/workspace/zinc_pretrain/Dataset/ZINC_smiles_dataset.smi"
    # save_filename = "/gpfs/wscgpfs02/chendo11/workspace/zinc_pretrain/Dataset/ZINC_smiles_dataset_canonical.smi"
