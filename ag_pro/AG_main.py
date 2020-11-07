# -*- coding: UTF-8 -*-
"""
Introduction:
    Algebraic Graph Learning
Author:
    Chend
Modify:
    2020-03-21
"""


# Depedence
import numpy as np
import glob
import sys
import os
import argparse
import time
from biopandas.mol2 import PandasMol2


class AlgebraicGraphFeatures(object):
    def __init__(self, kernal_type='Exponential', kernal_tau=3.5,
                 kernal_parameter=5.5, matrix_type='Laplacian'):

        self.cutoff = 12
        self.sigma = 0  # mean(std(ri),std(rk)) in dataset

        # Atomic radii
        self.atiomic_r = {
            'H': 0.53,
            'C': 0.67,
            'N': 0.56,
            'O': 0.48,
            'F': 0.42,
            'P': 0.98,
            'S': 0.87,
            'Cl': 0.79,
            'Br': 0.94,
            'I': 1.15,
        }

        # van der Waals radii
        self.van_der_waals_r = {
            'H': 1.2,
            'C': 1.77,
            'N': 1.66,
            'O': 1.5,
            'F': 1.46,
            'P': 1.9,
            'S': 1.89,
            'Cl': 1.82,
            'Br': 1.86,
            'I': 2.04,
        }

        self.statistic_eigen_value = 9
        self.kernal_type = kernal_type
        self.kernal_tau = kernal_tau
        self.kernal_parameter = kernal_parameter
        self.kernal_func = self.build_kernal_func(kernal_type)
        self.matrix_type = matrix_type

    def build_kernal_func(self, kernal_type):
        if kernal_type[0] in ['E', 'e']:
            return self.exponential_func
        elif kernal_type[0] in ['L', 'l']:
            return self.lorentz_func

    def exponential_func(self, atiomic_distance):
        eta = self.kernal_tau * (self.van_der_waals_r1 + self.van_der_waals_r2)
        phi = np.exp(-(atiomic_distance/eta) ** self.kernal_parameter)
        return np.round(phi, 5)

    def lorentz_func(self, atiomic_distance):
        eta = self.kernal_tau * (self.van_der_waals_r1 + self.van_der_waals_r2)
        phi = 1 / (1 + atiomic_distance/eta) ** self.kernal_parameter
        return np.round(phi, 5)

    def adjacency_matrix(self, cloudpoint1, cloudpoint2):
        length1 = cloudpoint1.shape[0]
        length2 = cloudpoint2.shape[0]

        graph_matrix = np.zeros((length1 + length2, length1 + length2))
        min_dis = self.r1 + self.r2 + self.sigma
        for i in range(length1 + length2):
            for j in range(i+1, length1 + length2):
                if j < length1 or i >= length1:
                    continue
                pair_points_dis = np.linalg.norm(
                    cloudpoint1[i] - cloudpoint2[j-length1])
                if pair_points_dis > min_dis and pair_points_dis <= self.cutoff:

                    # calculate kernal_function
                    phi = self.kernal_func(pair_points_dis)

                    # symmetric matrix
                    graph_matrix[i, j] = phi
                    graph_matrix[j, i] = phi

        return graph_matrix

    def laplacian_matrix(self, cloudpoint1, cloudpoint2):
        length1 = cloudpoint1.shape[0]
        length2 = cloudpoint2.shape[0]

        graph_matrix = np.zeros((length1 + length2, length1 + length2))
        min_dis = self.r1 + self.r2 + self.sigma
        for i in range(length1 + length2):
            for j in range(i+1, length1 + length2):
                if j < length1 or i >= length1:
                    continue
                pair_points_dis = np.linalg.norm(
                    cloudpoint1[i] - cloudpoint2[j-length1])
                if pair_points_dis > min_dis and pair_points_dis <= self.cutoff:
                    # calculate kernal_function
                    phi = self.kernal_func(pair_points_dis)

                    # symmetric matrix
                    graph_matrix[i, j] = -phi
                    graph_matrix[j, i] = -phi

            graph_matrix[i, i] = np.round(-np.sum(graph_matrix[:, i]), 5)

        return graph_matrix

    def graph_features(self, cloudpoint1, cloudpoint2, ele1, ele2):
        # protein and ligand atomic radius
        self.r1 = self.atiomic_r[ele1]
        self.r2 = self.atiomic_r[ele2]
        self.van_der_waals_r1 = self.van_der_waals_r[ele1]
        self.van_der_waals_r2 = self.van_der_waals_r[ele2]

        if self.matrix_type[0] in ['l', 'L']:
            # get Laplacian graph matrix
            graph_matrix = self.laplacian_matrix(cloudpoint1, cloudpoint2)

            # get eigen values and vectors
            eigen_values, eigen_vectors = np.linalg.eigh(graph_matrix)
            eigen_values = np.round(eigen_values, 5)
            eigen_vectors = np.round(eigen_vectors, 5)

            assert (eigen_values >= 0).all(), "Laplacian matrix's eigenvalues !< 0"

        elif self.matrix_type[0] in ['A', 'a']:
            # get Adjacency graph matrix
            graph_matrix = self.adjacency_matrix(cloudpoint1, cloudpoint2)

            # get eigen values and vectors
            eigen_values, eigen_vectors = np.linalg.eigh(graph_matrix)
            eigen_values = np.round(eigen_values, 5)
            eigen_vectors = np.round(eigen_vectors, 5)

            # reserve positive values
            eigen_values = np.array(list(filter(lambda v: v > 0, eigen_values)))

        if np.sum(np.abs(eigen_values)) == 0:
            agl_feature = np.zeros((9, ))
            return agl_feature

        # get statistic values
        positive_eigen_values = list(filter(lambda v: v > 0, eigen_values))
        agl_feature = [
            np.sum(positive_eigen_values),
            np.min(positive_eigen_values),  # Fiedler value for Laplacian matrices
            np.max(positive_eigen_values),
            np.mean(positive_eigen_values),
            np.median(positive_eigen_values),
            np.std(positive_eigen_values),
            np.var(positive_eigen_values),
            len(positive_eigen_values),
            np.sum(np.power(positive_eigen_values, 2))
        ]

        return np.round(agl_feature, 5)


def get_mol_feature(mol2file, agl_class):

    # get data
    solvation = PandasMol2().read_mol2(mol2file).df

    solvation_e = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']

    total_features = np.array([], dtype=float)
    for e1 in range(len(solvation_e)):
        ele1 = solvation_e[e1]
        for e2 in range(len(solvation_e)):
            # ligand's element
            ele2 = solvation_e[e2]
            cloudpoint1 = \
                solvation[['x', 'y', 'z']][solvation['atom_name'].str.contains('^' + ele1 + '[0-9]*$')].values
            cloudpoint2 = \
                solvation[['x', 'y', 'z']][solvation['atom_name'].str.contains('^' + ele2 + '[0-9]*$')].values

            if cloudpoint1.shape[0] == 0 or cloudpoint2.shape[0] == 0:
                # agl_features have 9 features
                agl_features = np.zeros((9, ))
            else:
                # each pair atoms feature
                agl_features = agl_class.graph_features(cloudpoint1, cloudpoint2, ele1, ele2)

            # store all pair features
            total_features = np.append(total_features, agl_features)

    return total_features


def main(args):
    kernal_type = args.kernal_type
    kernal_tau = args.kernal_tau
    kernal_parameter = args.kernal_parameter
    matrix_type = args.matrix_type
    dataset = args.dataset_path
    dataset_id = args.dataset_id_path

    # Data read, default data format is mol2
    files = glob.glob(os.path.join(dataset, '*.mol2'))
    print(f"{dataset} have {len(files)} files")

    # Get features function
    agl = AlgebraicGraphFeatures(kernal_type, kernal_tau, kernal_parameter, matrix_type)
    all_data_features = np.zeros((len(files), agl.statistic_eigen_value * 10 * 10), dtype=float)

    t1 = time.time()
    for i, id in enumerate(open(dataset_id)):
        file = os.path.join(dataset, id.strip() + '.mol2')
        features = get_mol_feature(file, agl)
        all_data_features[i, :] = features

    # save all feature
    save_feature_name = f"{args.dataset_prefix}_{matrix_type}_{kernal_type}_{kernal_parameter}_tau_{kernal_tau}"
    np.save(os.path.join(args.save_feature_path_prefix, save_feature_name), all_data_features)
    print(f'Running time: {time.time() - t1}')


def parse_args(args):
    parser = argparse.ArgumentParser(description='Get agl features for toxicity')

    parser.add_argument('--dataset_prefix', default='example_data', type=str,
                        help='The prefix filename of saved features.')
    parser.add_argument('--dataset_path', default='./examples/data/example_mol2', type=str,
                        help='The name of toxicity dataset folder')
    parser.add_argument('--dataset_id_path', default='./examples/data/example_stru.id', type=str,
                        help='Structure ids of dataset')
    parser.add_argument('--save_feature_path_prefix', default='./', type=str,
                        help='Folder of save_feature path')
    parser.add_argument('--matrix_type', default='Adj', type=str, choices=['Adj', 'Lap'])
    parser.add_argument('--kernal_type', default='Lorentz', type=str, choices=['Lorentz', 'Exponential'])
    parser.add_argument('--kernal_tau', default=0.5, type=float)
    parser.add_argument('--kernal_parameter', default=10.0, type=float)

    args = parser.parse_args()
    return args


def cli_main():
    args = parse_args(sys.argv[1:])
    print(args)
    main(args)


if __name__ == "__main__":
    cli_main()
    print('End!')
