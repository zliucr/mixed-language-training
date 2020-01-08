
from src.preprocess import load_woz_data
from src.utils import binarize_dst_data
from src.preparation import UNK_INDEX, PAD_INDEX

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

import numpy as np
import pickle

class Dataset(data.Dataset):
    def __init__(self, turns):
        self.turns = turns

    def __getitem__(self, index):
        # dialog_idx, utterance, acts_request, acts_slot_type, acts_slot_value, turn_slot, turn_slot_label, turn_request_label
        return self.turns[index][0], self.turns[index][1], self.turns[index][2], self.turns[index][3], self.turns[index][4], self.turns[index][5], self.turns[index][6], self.turns[index][7]
    
    def __len__(self):
        return len(self.turns)

def collate_fn(data):
    dialogu_idx, utterances, acts_request, acts_slot, acts_values, slot_names, turn_slot_labels, turn_request_labels = zip(*data)
    
    lengths = [ len(utter) for utter in utterances ]
    max_lengths = max(lengths)
    padded_seqs = torch.LongTensor(len(utterances), max_lengths).fill_(PAD_INDEX)
    for i, seq in enumerate(utterances):
        length = lengths[i]
        padded_seqs[i, :length] = torch.LongTensor(seq)
    lengths = torch.LongTensor(lengths)
    
    turn_slot_labels = torch.LongTensor(turn_slot_labels)
    turn_request_labels = torch.FloatTensor(turn_request_labels)

    return dialogu_idx, padded_seqs, lengths, acts_request, acts_slot, acts_values, slot_names, turn_slot_labels, turn_request_labels

def load_data(params, dialogue_ontology, mapping=None):
    pri_turns_train = load_woz_data("data/dst/dst_data/tok_woz_train_en.json", "en", dialogue_ontology, mapping=mapping)
    pri_turns_val = load_woz_data("data/dst/dst_data/tok_woz_validate_en.json", "en", dialogue_ontology, mapping=mapping)
    val_count = len(pri_turns_val)
    pri_turns_train = pri_turns_train + pri_turns_val[0:int(0.75 * val_count)]

    tgt_pri_turns_val = load_woz_data("data/dst/dst_data/tok_woz_validate_"+params.trans_lang+".json", params.trans_lang, dialogue_ontology, mapping=mapping)
    tgt_pri_turns_test = load_woz_data("data/dst/dst_data/tok_woz_test_"+params.trans_lang+".json", params.trans_lang, dialogue_ontology, mapping=mapping)
    
    return pri_turns_train, tgt_pri_turns_val, tgt_pri_turns_test

def get_dst_dataloader(params, vocab_en, vocab_trans, dialogue_ontology):
    if params.mix_train == True:
        # load mapping for mix training
        with open(params.mapping_for_mix, "rb") as f:
            mapping_for_mix = pickle.load(f)
    else:
        mapping_for_mix = None
    train_turns, tgt_val_turns, tgt_test_turns = load_data(params, dialogue_ontology, mapping=mapping_for_mix)

    train_turns_bin = binarize_dst_data(params, train_turns, vocab_en, dialogue_ontology, lang="en", isTestset=False)
    tgt_turns_val_bin = binarize_dst_data(params, tgt_val_turns, vocab_trans, dialogue_ontology, lang=params.trans_lang, isTestset=True)
    tgt_turns_test_bin = binarize_dst_data(params, tgt_test_turns, vocab_trans, dialogue_ontology, lang=params.trans_lang, isTestset=True)

    dataset_tr = Dataset(train_turns_bin)
    dataset_val = Dataset(tgt_turns_val_bin)
    dataset_test = Dataset(tgt_turns_test_bin)

    dataloader_tr = DataLoader(dataset=dataset_tr, batch_size=params.batch_size, shuffle=True, collate_fn=collate_fn)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=params.batch_size, shuffle=False, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=params.batch_size, shuffle=False, collate_fn=collate_fn)

    return dataloader_tr, dataloader_val, dataloader_test

