# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from MolecularAI/MolBART
#
# Source:
# https://github.com/MolecularAI/MolBART/blob/master/molbart/util.py
#
# The license for the original version of this file can be
# found in this directory (LICENSE_MOLBART).
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------

# coding=utf-8

import os

project_home = os.environ['PROJECT_HOME']

root = project_home
DEFAULT_VOCAB_PATH = os.path.join(root, '/home/aih/valentin.rauscher/Jointformer_RetMol_experiment/RetMol/MolBART/megatron_molbart/jointformer/data/vocabularies/deepchem.txt')
CHECKPOINTS_DIR = os.path.join(root, 'models/jointformer/')

# Tokenization and vocabulary
DEFAULT_MAX_SEQ_LEN = 128
DEFAULT_CHEM_TOKEN_START = 15
DEFAULT_BEGIN_TOKEN = "[CLS]"
DEFAULT_END_TOKEN = "[SEP]"
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_UNK_TOKEN = "[UNK]"
DEFAULT_MASK_TOKEN = "[MASK]"
DEFAULT_SEP_TOKEN = "[SEP]"
DEFAULT_MASK_PROB = 0.15
DEFAULT_SHOW_MASK_TOKEN_PROB = 1.0
DEFAULT_MASK_SCHEME = "span"
DEFAULT_SPAN_LAMBDA = 3.0
REGEX = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#||\+|\\\\\/|:||@|\?|>|\*|\$|\%[0–9]{2}|[0–9])"

# Model parameters
DEFAULT_D_MODEL = 256
DEFAULT_NUM_LAYERS = 8
DEFAULT_NUM_HEADS = 8
