# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NSCL license
# for RetMol. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

# coding=utf-8

import os, sys
import pandas as pd
import pickle as pkl
import random
import torch
from torch.utils.data import Dataset
from pysmilesutils.augment import SMILESAugmenter
from megatron_molbart.util_jointformer import DEFAULT_CHEM_TOKEN_START, DEFAULT_VOCAB_PATH, DEFAULT_MAX_SEQ_LEN, REGEX
from utils.tokenizer import load_tokenizer

project_home = os.environ['PROJECT_HOME']
sys.path.insert(1, os.path.join(project_home, 'MolBART/megatron_molbart/Megatron-LM-v1.1.5-3D_parallelism'))
from megatron.data.samplers import DistributedBatchSampler
from megatron import mpu
sys.path.insert(1, os.path.join(project_home, 'MolBART/megatron_molbart/'))
from jointformer.configs.tokenizer import TokenizerConfig
from jointformer.utils.tokenizers.auto import AutoTokenizer



random.seed(123)
project_home = os.environ['PROJECT_HOME']
### supply the path to the config of the tokenizer you intend to use
PATH_TO_TOKENIZER_CONFIG='_/path/to/tokenizer/config/_'
tokenizer_config = TokenizerConfig.from_config_file(PATH_TO_TOKENIZER_CONFIG)
default_tokenizer = AutoTokenizer.from_config(tokenizer_config)


def check_seq_len(tokens, mask, max_seq_len=DEFAULT_MAX_SEQ_LEN):
    """ Warn user and shorten sequence if the tokens are too long, otherwise return original

    Args:
        tokens (List[List[str]]): List of token sequences
        mask (List[List[int]]): List of mask sequences

    Returns:
        tokens (List[List[str]]): List of token sequences (shortened, if necessary)
        mask (List[List[int]]): List of mask sequences (shortened, if necessary)
    """

    seq_len = max([len(ts) for ts in tokens])
    if seq_len > max_seq_len:
        tokens_short = [ts[:max_seq_len] for ts in tokens]
        mask_short = [ms[:max_seq_len] for ms in mask]
        return (tokens_short, mask_short)
    return (tokens, mask)


def collate_fn(batch):
    """ Used by DataLoader to concatenate/collate inputs."""

    encoder_smiles = [x['encoder_smiles'] for x in batch]
    decoder_smiles = [x['decoder_smiles'] for x in batch]
    mask = batch[0]['train']

    enc_token_output = default_tokenizer(encoder_smiles,"prediction")
    dec_token_output = default_tokenizer(decoder_smiles, "prediction")
    enc_token_ids = enc_token_output['input_ids'].transpose(0,1)
    enc_att_mask = enc_token_output['attention_mask'].transpose(0,1)
    dec_token_ids = dec_token_output['input_ids'].transpose(0,1)
    target_pad_mask = (enc_token_ids == 0)
    # process the retrieved molecule smiles
    retrieved_smiles = [x['retrieved_smiles'] for x in batch]  # this is a list of lists
    retrieved_smiles = [item for sublist in retrieved_smiles
                        for item in sublist]  # this is a list of blocks of batch_size, each block size = #retrievals
    ret_token_output = default_tokenizer(retrieved_smiles, "prediction")  # dim = b*k, r
    ret_tokens = ret_token_output['input_ids']
    ret_mask = ret_token_output['attention_mask']
    ret_token_ids = ret_tokens
    ret_token_ids = torch.tensor(ret_token_ids).transpose(0, 1)  # dim=r, b*k
    ret_token_ids = ret_token_ids.reshape(ret_token_ids.shape[0], len(batch),
                                          len(batch[0]['retrieved_smiles']))  # dim = r, b, k
    ret_token_ids = torch.movedim(ret_token_ids, -1, 0).transpose(0,1)  # dim = k, r, b
    ret_pad_mask = torch.tensor(ret_mask, dtype=torch.int64)  # dim = b*k, r
    ret_pad_mask = ret_pad_mask.reshape(len(batch), len(batch[0]['retrieved_smiles']),
                                        ret_pad_mask.shape[-1])  # dim = b, k, r
    ret_pad_mask = ret_pad_mask.reshape(len(batch),
                                        len(batch[0]['retrieved_smiles']) * ret_pad_mask.shape[-1])  # dim = b, k*r
    #######################################



    collate_output = {
        'encoder_input': enc_token_ids,
        'encoder_pad_mask': enc_att_mask,
        'target': dec_token_ids[1:, :],
        'target_pad_mask': target_pad_mask[1:, :],
        'encoder_smiles': encoder_smiles,
        'decoder_smiles': decoder_smiles,
        'retrieved_smiles': ret_token_ids,
        'retrieved_pad_mask': ret_pad_mask
    }

    return collate_output


def collate_fn_inference(batch):
    """ Used by DataLoader to concatenate/collate inputs."""

    encoder_smiles = [x['encoder_smiles'] for x in batch]

    enc_token_output = default_tokenizer(encoder_smiles,"generation")
    enc_token_ids = enc_token_output['input_ids'].transpose(0,1)
    enc_att_mask = enc_token_output['attention_mask'].transpose(0,1)
    target_pad_mask = (enc_token_ids == 0)
    # process the retrieved molecule smiles
    retrieved_smiles = [x['retrieved_smiles'] for x in batch]  # this is a list of lists
    retrieved_smiles = [item for sublist in retrieved_smiles
                        for item in sublist]  # this is a list of blocks of batch_size, each block size = #retrievals
    ret_token_output = default_tokenizer(retrieved_smiles, "generation")  # dim = b*k, r
    ret_tokens = ret_token_output['input_ids']
    ret_mask = ret_token_output['attention_mask']
    ret_token_ids = ret_tokens
    ret_token_ids = torch.tensor(ret_token_ids).transpose(0, 1)  # dim=r, b*k
    ret_token_ids = ret_token_ids.reshape(ret_token_ids.shape[0], len(batch),
                                          len(batch[0]['retrieved_smiles']))  # dim = r, b, k
    ret_token_ids = torch.movedim(ret_token_ids, -1, 0).transpose(0,1)  # dim = k, r, b
    ret_pad_mask = torch.tensor(ret_mask, dtype=torch.int64)  # dim = b*k, r
    ret_pad_mask = ret_pad_mask.reshape(len(batch), len(batch[0]['retrieved_smiles']),
                                        ret_pad_mask.shape[-1])  # dim = b, k, r
    ret_pad_mask = ret_pad_mask.reshape(len(batch),
                                        len(batch[0]['retrieved_smiles']) * ret_pad_mask.shape[-1])  # dim = b, k*r
    #######################################
    collate_output = {
        'encoder_input': enc_token_ids.cuda(),
        'encoder_pad_mask': enc_att_mask.cuda(),
        'retrieved_smiles': ret_token_ids.cuda(),
        'retrieved_pad_mask': ret_pad_mask.cuda()
    }
    return collate_output


def collate_fn_retrieved_only(retrieved_smiles, batch_size, n_retrievals):
    ret_token_output = default_tokenizer.tokenize(retrieved_smiles, pad=True)  # dim = b*k, r
    ret_tokens = ret_token_output['original_tokens']
    ret_mask = ret_token_output['original_pad_masks']
    (ret_tokens, ret_mask) = check_seq_len(ret_tokens, ret_mask)
    ret_token_ids = default_tokenizer.convert_tokens_to_ids(ret_tokens)
    ret_token_ids = torch.tensor(ret_token_ids).transpose(0, 1)  # dim=r, b*k
    ret_token_ids = ret_token_ids.reshape(ret_token_ids.shape[0], batch_size, n_retrievals)  # dim = r, b, k
    ret_token_ids = torch.movedim(ret_token_ids, -1, 0)  # dim = k, r, b
    ret_pad_mask = torch.tensor(ret_mask, dtype=torch.int64)  # dim = b*k, r
    ret_pad_mask = ret_pad_mask.reshape(batch_size, n_retrievals, ret_pad_mask.shape[-1])  # dim = b, k, r
    ret_pad_mask = ret_pad_mask.reshape(batch_size, n_retrievals * ret_pad_mask.shape[-1])  # dim = b, k*r
    return ret_token_ids.cuda(), ret_pad_mask.cuda()


class MoleculeDataset(Dataset):
    """Simple Molecule dataset that reads from a single DataFrame."""

    def __init__(self, df, retrieval, n_retrieval, myargs, train=True):
        """
        Args:
            df (pandas.DataFrame): DataFrame object with RDKit molecules and lengths.
            retrieval (python dict): dictionary of retrieved molecules for each molecule in the zinc250k dataset
        """
        self.mols = df.smiles.tolist()
        self.retrieval = retrieval
        self.n_retrieval = n_retrieval
        self.myargs = myargs
        self.aug = SMILESAugmenter()
        self.train = train

        # for attr only
        tmp_set = set(self.mols)
        tmp_set2 = set(self.retrieval.keys())
        for item in tmp_set:
            if item not in tmp_set2:
                self.mols.remove(item)

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        mol = self.mols[idx]
        try:
            if self.train:
                ## augmentation
                if self.myargs.enumeration_input:
                    enc_smi = self.aug(mol)
                    assert (mol in enc_smi)
                else:
                    ## no augmentation
                    enc_smi = [mol]
            else:
                enc_smi = [mol]
        except:
            enc_smi = [mol]

        # get the encoder and decoder smiles
        encoder_smiles = random.choice(enc_smi)
        if self.myargs.pred_target == 'reconstruction':
            decoder_smiles = encoder_smiles
        elif self.myargs.pred_target == 'canonical':
            decoder_smiles = mol
        elif self.myargs.pred_target == 'nearestn':
            start_idx = 1 if self.retrieval[mol][0] == mol else 0
            decoder_smiles = self.retrieval[mol][start_idx]
            self.retrieval[mol] = self.retrieval[mol][start_idx + 1:]

        # get the retrieved molecules  
        start_idx = 1 if self.retrieval[mol][0] == mol else 0
        if self.myargs.retriever_rule == 'topk':
            retrieved = self.retrieval[mol][start_idx:self.n_retrieval + start_idx]
        elif self.myargs.retriever_rule == 'random':
            retrieved = self.retrieval[mol][start_idx:self.myargs.n_neighbors + start_idx]
            retrieved = random.sample(retrieved, self.n_retrieval)
            retrieved = self.retrieval[mol][start_idx:self.n_retrieval + start_idx]
        elif self.myargs.retriever_rule == 'structrand':
            retrieved = self.retrieval[mol][start_idx:self.myargs.n_neighbors + start_idx]
            retrieved = random.sample(retrieved, self.n_retrieval + 1)
            decoder_smiles = random.sample(retrieved, 1)
            assert (len(decoder_smiles) == 1)
            decoder_smiles = decoder_smiles[0]
            retrieved.remove(decoder_smiles)

        assert (len(retrieved) == self.n_retrieval)

        output = {'encoder_smiles': encoder_smiles,
                  'decoder_smiles': decoder_smiles,
                  'retrieved_smiles': retrieved,
                  'train': self.train}
        return output


class MoleculeDataLoader(object):
    """Loads data from a csv file containing molecules."""

    def __init__(
            self,
            file_path,
            myargs,
            batch_size=32,
            num_buckets=20,
            num_workers=32,
            vocab_path=DEFAULT_VOCAB_PATH,
            chem_token_start=DEFAULT_CHEM_TOKEN_START,
            regex=REGEX,
            train=True
    ):

        from tdc.generation import MolGen
        data = MolGen(name='ZINC', path=os.path.join(project_home, 'data/zinc'))
        splits = data.get_split()

        if myargs.stage == 1:
            retrieval_set_train = pkl.load(open(os.path.join(
                project_home,
                'data/retrieval-precompute/similarity_precompute/'
                'knn100_zinc250_cddd_scann_normalized_TrainSetRetrieval_TrainSetQuery.pkl'),
                'rb'))  # stage 1, tdc data
            retrieval_set_valid = pkl.load(open(os.path.join(
                project_home,
                'data/retrieval-precompute/similarity_precompute/'
                'knn100_zinc250_cddd_scann_normalized_TrainSetRetrieval_ValidSetQuery.pkl'),
                'rb'))  # stage 1, tdc data
            retrieval_set_test = pkl.load(open(os.path.join(
                project_home,
                'data/retrieval-precompute/similarity_precompute/'
                'knn100_zinc250_cddd_scann_normalized_TrainSetRetrieval_TestSetQuery.pkl'),
                'rb'))  # stage 1, tdc data

        elif myargs.stage == 2:
            attr_offset = int(myargs.attr_offset) if int(
                myargs.attr_offset) == myargs.attr_offset else myargs.attr_offset
            retrieval_set_train = pkl.load(open(os.path.join(
                project_home,
                'data/retrieval-precompute/similarity_{}_precompute/'
                'knn{}_zinc250_cddd_scann_normalized_TrainSetRetrieval_TrainSetQuery_offset{}.pkl'.format(
                    myargs.attr, myargs.n_neighbors, attr_offset)),
                'rb'))  # stage 2
            retrieval_set_valid = pkl.load(open(os.path.join(
                project_home,
                'data/retrieval-precompute/similarity_{}_precompute/'
                'knn{}_zinc250_cddd_scann_normalized_TrainSetRetrieval_ValidSetQuery_offset{}.pkl'.format(
                    myargs.attr, myargs.n_neighbors, attr_offset)),
                'rb'))  # stage 2
            retrieval_set_test = pkl.load(open(os.path.join(
                project_home,
                'data/retrieval-precompute/similarity_{}_precompute/'
                'knn{}_zinc250_cddd_scann_normalized_TrainSetRetrieval_TestSetQuery_offset{}.pkl'.format(
                    myargs.attr, myargs.n_neighbors, attr_offset)),
                'rb'))  # stage 2

        if myargs.data_source == 'jtnn':
            retrievals_all = {**retrieval_set_train, **retrieval_set_valid, **retrieval_set_test}
            train = open(os.path.join(project_home, 'data/zinc/train.txt')).readlines()
            train = [item.strip() for item in train]
            train = list(set(train).intersection(set(retrievals_all.keys())))
            valid = open(os.path.join(project_home, 'data/zinc/valid.txt')).readlines()
            valid = [item.strip() for item in valid]
            valid = list(set(valid).intersection(set(retrievals_all.keys())))
            test = open(os.path.join(project_home, 'data/zinc/test.txt')).readlines()
            test = [item.strip() for item in test]
            test = list(set(test).intersection(set(retrievals_all.keys())))
            splits['train'] = pd.DataFrame({'smiles': train})
            splits['valid'] = pd.DataFrame({'smiles': valid})
            splits['test'] = pd.DataFrame({'smiles': test})
            retrieval_set_train = {mol: retrievals_all[mol] for mol in train}
            retrieval_set_valid = {mol: retrievals_all[mol] for mol in valid}
            retrieval_set_test = {mol: retrievals_all[mol] for mol in test}

        train_dataset = MoleculeDataset(splits['train'], retrieval_set_train, myargs.n_retrievals, myargs, train=True)
        val_dataset = MoleculeDataset(splits['valid'], retrieval_set_valid, myargs.n_retrievals, myargs, train=False)
        test_dataset = MoleculeDataset(splits['test'], retrieval_set_test, myargs.n_retrievals, myargs, train=False)
        self.tokenizer = default_tokenizer

        world_size = torch.distributed.get_world_size(group=mpu.get_data_parallel_group())
        rank = torch.distributed.get_rank(group=mpu.get_data_parallel_group())

        sampler = torch.utils.data.SequentialSampler(train_dataset)
        batch_sampler = DistributedBatchSampler(sampler, batch_size,
                                                True, rank, world_size)
        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_sampler=batch_sampler, num_workers=num_workers,
                                                        pin_memory=True, collate_fn=collate_fn)

        sampler = torch.utils.data.SequentialSampler(val_dataset)
        batch_sampler = DistributedBatchSampler(sampler, batch_size,
                                                True, rank, world_size)
        self.val_loader = torch.utils.data.DataLoader(val_dataset,
                                                      batch_sampler=batch_sampler,
                                                      num_workers=num_workers, pin_memory=True,
                                                      collate_fn=collate_fn)

        sampler = torch.utils.data.SequentialSampler(test_dataset)
        batch_sampler = DistributedBatchSampler(sampler, batch_size,
                                                True, rank, world_size)
        self.test_loader = torch.utils.data.DataLoader(test_dataset,
                                                       batch_sampler=batch_sampler,
                                                       num_workers=num_workers, pin_memory=True,
                                                       collate_fn=collate_fn)

    def get_data(self):
        return (self.train_loader, self.val_loader)

    def get_data_test(self):
        return (self.train_loader, self.val_loader, self.test_loader)
