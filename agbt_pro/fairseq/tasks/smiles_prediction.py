# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np

from fairseq.data import (
    data_utils,
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumSamplesDataset,
    NumelDataset,
    OffsetTokensDataset,
    PrependTokenDataset,
    RawLabelDataset,
    RightPadDataset,
    RollDataset,
    SortDataset,
    StripTokenDataset,
    TruncateDataset,
)

from . import FairseqTask, register_task


@register_task('smiles_prediction')
class SmilesPredictionTask(FairseqTask):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', metavar='FILE',
                            help='file prefix for data')
        parser.add_argument('--dict_path', type=str, default=None,
                            help='Source dict.txt path')
        parser.add_argument('--num-classes', type=int, default=-1,
                            help='number of classes')
        parser.add_argument('--init-token', type=int, default=None,
                            help='add token at the beginning of each batch item')
        parser.add_argument('--separator-token', type=int, default=None,
                            help='add separator token between inputs')
        parser.add_argument('--regression-target', action='store_true', default=False)
        parser.add_argument('--no-shuffle', action='store_true', default=False)
        parser.add_argument('--truncate-sequence', action='store_true', default=False,
                            help='Truncate sequence to max_sequence_length')
        parser.add_argument('--add-prev-output-tokens', action='store_true', default=False,
                            help='Add prev_output_tokens to sample, used for encoder-decoder arch')

    def __init__(self, args, data_dictionary, label_dictionary):
        super().__init__(args)
        self.dictionary = data_dictionary
        self._label_dictionary = label_dictionary
        if not hasattr(args, 'max_positions'):
            self._max_positions = (
                args.max_source_positions,
                args.max_target_positions)
        else:
            self._max_positions = args.max_positions
        args.tokens_per_sample = self._max_positions

    @classmethod
    def load_dictionary(cls, args, filename, source=True):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        dictionary.add_symbol('<mask>')
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert args.num_classes > 0, 'Must set --num-classes'

        # load data dictionary
        if args.dict_path is None:
            dict_path = os.path.join(args.data, 'dict.txt')
        else:
            dict_path = args.dict_path
        assert os.path.exists(dict_path), f"dict.txt doesn't exist in {args.data}"
        data_dict = cls.load_dictionary(args, dict_path, source=True)
        print('| [input] dictionary: {} types'.format(len(data_dict)))

        label_dict = None
        if not args.regression_target:
            # load label dictionary
            label_dict = cls.load_dictionary(
                args,
                os.path.join(args.data, 'label', 'dict.txt'),
                source=False)
            print('| [label] dictionary: {} types'.format(len(label_dict)))
        else:
            label_dict = data_dict
        return SmilesPredictionTask(args, data_dict, label_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""
        def get_path(split):
            return os.path.join(self.args.data, split)

        def make_dataset(split_path, dictionary):
            dataset = data_utils.load_indexed_dataset(
                split_path,
                self.source_dictionary,
                self.args.dataset_impl,
                combine=combine)
            return dataset

        input0 = make_dataset(
            os.path.join(self.args.data, split), self.source_dictionary)
        assert input0 is not None, 'could not find dataset: {}'.format(
            os.path.join(self.args.data, split))

        if self.args.init_token is not None:
            input0 = PrependTokenDataset(input0, self.args.init_token)

        src_tokens = input0

        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(src_tokens))

        if self.args.truncate_sequence:
            src_tokens = TruncateDataset(src_tokens, self.args.max_positions)

        dataset = {
            'id': IdDataset(),
            'net_input': {
                'src_tokens': RightPadDataset(
                    src_tokens,
                    pad_idx=self.source_dictionary.pad()),
                'src_lengths': NumelDataset(src_tokens, reduce=False)},
            'nsentences': NumSamplesDataset(),
            'ntokens': NumelDataset(src_tokens, reduce=True),
        }

        if self.args.add_prev_output_tokens:
            prev_tokens_dataset = RightPadDataset(
                RollDataset(src_tokens, 1),
                pad_idx=self.dictionary.pad())
            dataset['net_input'].update(
                prev_output_tokens=prev_tokens_dataset)

        if not self.args.regression_target:
            label_dataset = make_dataset(
                os.path.join(self.args.data, split + '.label'), self.target_dictionary)
            if label_dataset is not None:
                dataset.update(
                    target=OffsetTokensDataset(
                        StripTokenDataset(
                            label_dataset,
                            id_to_strip=self.target_dictionary.eos()),
                        offset=-self.target_dictionary.nspecial,
                    )
                )
        else:
            label_path = os.path.join(self.args.data, split + '.label')
            if os.path.exists(label_path):
                dataset.update(target=RawLabelDataset([
                    float(x.strip()) for x in open(label_path).readlines()]))

        nested_dataset = NestedDictionaryDataset(dataset, sizes=[src_tokens.sizes])

        if self.args.no_shuffle:
            dataset = nested_dataset
        else:
            dataset = SortDataset(
                nested_dataset,
                sort_order=[shuffle])  # shuffle

        print("| Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)

        # model.register_classification_head('sentence_classification_head',
        #                                    num_classes=self.args.num_classes)
        model.register_pooler_regression_head('sentence_classification_head',
                                              num_classes=self.args.num_classes)
        return model

    def max_positions(self):
        return self._max_positions

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    @property
    def label_dictionary(self):
        return self._label_dictionary
