
import torch
from torch import nn
from torch.nn import functional as F
from src.dst_model import Attention
from src.crf import *
from src.utils import load_embedding
from src.preparation import PAD_INDEX

SLOT_PAD = 0

class Lstm_nlu(nn.Module):
    def __init__(self, params, vocab_en, vocab_trans):
        super(Lstm_nlu, self).__init__()
        self.n_layer = params.n_layer
        self.emb_dim = params.emb_dim
        self.hidden_dim = params.hidden_dim
        self.dropout = params.dropout
        self.bidirection = params.bidirection
        
        self.emb_file_en = params.emb_file_en
        self.emb_file_trans = params.emb_file_trans
        self.n_words_en = vocab_en.n_words
        self.n_words_trans = vocab_trans.n_words

        # English embedding layer
        self.embedding_en = nn.Embedding(self.n_words_en, self.emb_dim, padding_idx=PAD_INDEX)
        # load embedding
        embedding_en = load_embedding(self.emb_file_en)
        self.embedding_en.weight.data.copy_(torch.FloatTensor(embedding_en))
        
        # Transfer language embeddings layer
        self.embedding_trans = nn.Embedding(self.n_words_trans, self.emb_dim, padding_idx=PAD_INDEX)
        # load embedding
        embedding_trans = load_embedding(self.emb_file_trans)
        self.embedding_trans.weight.data.copy_(torch.FloatTensor(embedding_trans))

        # LSTM layers
        self.lstm = nn.LSTM(self.emb_dim, self.hidden_dim, num_layers=self.n_layer, dropout=self.dropout, bidirectional=self.bidirection, batch_first=True)
    
    def forward(self, x, lang):
        """
        Input:
            x: text (bsz, seq_len)
        Output:
            lstm_output: (bsz, seq_len, hidden_dim)
        """
        embeddings = self.embedding_en(x) if lang == "en" else self.embedding_trans(x)
        embeddings = embeddings.detach()
        embeddings = F.dropout(embeddings, p=self.dropout, training=self.training)
        
        # LSTM
        # lstm_output (batch_first): (bsz, seq_len, hidden_dim)
        lstm_output, (_, _) = self.lstm(embeddings)

        return lstm_output


class IntentPredictor(nn.Module):
    def __init__(self, params):
        super(IntentPredictor, self).__init__()
        self.num_intent = params.num_intent
        self.attention_size = params.hidden_dim * 2 if params.bidirection else params.hidden_dim
        self.atten_layer = Attention(attention_size=self.attention_size, return_attention=False)
        
        self.linear = nn.Linear(self.attention_size, self.num_intent)
        
    def forward(self, inputs, lengths):
        """ forward pass
        Inputs:
            inputs: lstm hidden layer (bsz, seq_len, hidden_dim)
            lengths: lengths of x (bsz, )
        Output:
            prediction: Intent prediction (bsz, num_intent)
        """
        atten_layer, _ = self.atten_layer(inputs, lengths)
        prediction = self.linear(atten_layer)

        return prediction

class SlotPredictor(nn.Module):
    def __init__(self, params):
        super(SlotPredictor, self).__init__()
        self.num_slot = params.num_slot
        self.hidden_dim = params.hidden_dim * 2 if params.bidirection else params.hidden_dim
        
        self.linear = nn.Linear(self.hidden_dim, self.num_slot)
        self.crf_layer = CRF(self.num_slot)
    
    def forward(self, inputs):
        """ forward pass
        Input:
            inputs: lstm hidden layer (bsz, seq_len, hidden_dim)
        Output:
            prediction: slot prediction (bsz, seq_len, num_slot)
        """
        prediction = self.linear(inputs)
        return prediction

    def out_vae_layer(self, inputs):
        assert self.vae == True
        vae_layer, _ = self.vae_layer(inputs)
        return vae_layer

    def crf_loss(self, inputs, lengths, y):
        """ create crf loss
        Input:
            inputs: output of SlotPredictor (bsz, seq_len, num_slot)
            lengths: lengths of x (bsz, )
            y: label of slot value (bsz, seq_len)
        Ouput:
            crf_loss: loss of crf
        """
        padded_y = self.pad_label(lengths, y)
        mask = self.make_mask(lengths)
        crf_loss = self.crf_layer.loss(inputs, padded_y)

        return crf_loss
    
    def crf_decode(self, inputs, lengths):
        """ crf decode
        Input:
            inputs: output of SlotPredictor (bsz, seq_len, num_slot)
            lengths: lengths of x (bsz, )
        Ouput:
            crf_loss: loss of crf
        """
        mask = self.make_mask(lengths)
        prediction = self.crf_layer(inputs)
        prediction = [ prediction[i, :length].data.cpu().numpy() for i, length in enumerate(lengths) ]

        return prediction

    def make_mask(self, lengths):
        bsz = len(lengths)
        max_len = torch.max(lengths)
        mask = torch.LongTensor(bsz, max_len).fill_(1)
        for i in range(bsz):
            length = lengths[i]
            mask[i, length:max_len] = 0
        mask = mask.cuda()
        return mask

    def pad_label(self, lengths, y):
        bsz = len(lengths)
        max_len = torch.max(lengths)
        padded_y = torch.LongTensor(bsz, max_len).fill_(SLOT_PAD)
        for i in range(bsz):
            length = lengths[i]
            y_i = y[i]
            padded_y[i, 0:length] = torch.LongTensor(y_i)

        padded_y = padded_y.cuda()
        return padded_y
