# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from fairseq.data.encoders import register_bpe
from fairseq.tokenizer import tokenize_smiles
from fairseq.data import Dictionary


@register_bpe('smi')
class SMI2BPE(object):
    def __init__(self, args):
        dict_file = os.path.join(args.data, 'dict.txt')
        assert os.path.exists(dict_file), f"dict.txt doesn't exists in {args.data}"
        self.vocab_dict = Dictionary.load(dict_file)

    def encode(self, x):
        return ' '.join(tokenize_smiles(x))

    def decode(self, x):
        return x

    def is_beginning_of_word(self, x: str) -> bool:
        return self.decode(x).startswith(' ')
