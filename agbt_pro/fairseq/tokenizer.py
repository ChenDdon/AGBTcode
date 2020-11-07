# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re

SPACE_NORMALIZER = re.compile(r"\s+")
SMI_SYMBOLS = r"Li|Be|Na|Mg|Al|Si|Cl|Ca|Zn|As|Se|se|Br|Rb|Sr|Ag|Sn|Te|te|Cs|Ba|Bi|[\d]|" + \
                r"[HBCNOFPSKIbcnops#%\)\(\+\-\\\/\.=@\[\]]"


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


def tokenize_smiles(line):
    line = re.findall(SMI_SYMBOLS, line.strip())
    return line


if __name__ == "__main__":
    print('End!')
