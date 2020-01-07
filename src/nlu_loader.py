
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from src.preprocess import preprocess_nlu_data
from src.preparation import PAD_INDEX

import pickle
import logging
logger = logging.getLogger()

class Dataset(data.Dataset):
    def __init__(self, data):
        self.X = data["text"]
        self.y1 = data["intent"]
        self.y2 = data["slot"]

    def __getitem__(self, index):
        return self.X[index], self.y1[index], self.y2[index] 
    
    def __len__(self):
        return len(self.X)

def collate_fn(data):
    X, y1, y2 = zip(*data)
    lengths = [len(bs_x) for bs_x in X]
    max_lengths = max(lengths)
    padded_seqs = torch.LongTensor(len(X), max_lengths).fill_(PAD_INDEX)
    for i, seq in enumerate(X):
        length = lengths[i]
        padded_seqs[i, :length] = torch.LongTensor(seq)
    lengths = torch.LongTensor(lengths)
    y1 = torch.LongTensor(y1)
    return padded_seqs, lengths, y1, y2

def load_data(params):
    data = {"en": {}, "es": {}, "th": {}}
    if params.mix_train == True:
        # load mapping for mix training
        with open(params.mapping_for_mix, "rb") as f:
            token_mapping = pickle.load(f)
    else:
        token_mapping = None
    # load English data
    preprocess_nlu_data(data, "en", params.clean_txt, token_mapping=token_mapping, vocab_path=params.vocab_path_en)
    # load Transfer language data
    preprocess_nlu_data(data, params.trans_lang, params.clean_txt, vocab_path=params.vocab_path_trans, filtered=params.filtered, filtered_scale=params.filter_scale)

    return data

def get_nlu_dataloader(params):
    data = load_data(params)

    dataset_tr = Dataset(data["en"]["train"])
    dataset_val = Dataset(data[params.trans_lang]["eval"])
    dataset_test = Dataset(data[params.trans_lang]["test"])

    dataloader_tr = DataLoader(dataset=dataset_tr, batch_size=params.batch_size, shuffle=True, collate_fn=collate_fn)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=params.batch_size, shuffle=False, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=params.batch_size, shuffle=False, collate_fn=collate_fn)
    
    return dataloader_tr, dataloader_val, dataloader_test, data["en"]["vocab"], data[params.trans_lang]["vocab"]
