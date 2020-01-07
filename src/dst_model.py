
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable

from src.utils import load_embedding, load_ontology_embedding
from src.preparation import PAD_INDEX

class Lstm(nn.Module):
    def __init__(self, params, vocab_en, vocab_trans):
        super(Lstm, self).__init__()
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
        # load English embedding
        embedding_en = load_embedding(self.emb_file_en)
        self.embedding_en.weight.data.copy_(torch.FloatTensor(embedding_en))

        # Transfer language embeddings
        self.embedding_trans = nn.Embedding(self.n_words_trans, self.emb_dim, padding_idx=PAD_INDEX)
        # load transfer language embedding
        embedding_trans = load_embedding(self.emb_file_trans)
        self.embedding_trans.weight.data.copy_(torch.FloatTensor(embedding_trans))
        
        # LSTM layers
        self.lstm = nn.LSTM(self.emb_dim, self.hidden_dim, dropout=self.dropout, bidirectional=self.bidirection, batch_first=True)
    
    def forward(self, x, lang):
        """
        Input:
            x: text (bsz, seq_len)
        Output:
            last_layer: last layer of lstm (bsz, seq_len, hidden_dim)
        """
        embeddings = self.embedding_en(x) if lang == "en" else self.embedding_trans(x)
        embeddings = embeddings.detach()
        embeddings = F.dropout(embeddings, p=self.dropout, training=self.training)

        # lstm_output (batch_first): (bsz, seq_len, hidden_dim)
        lstm_output, (_, _) = self.lstm(embeddings)

        return lstm_output

# https://github.com/huggingface/torchMoji/blob/master/torchmoji/attlayer.py
class Attention(nn.Module):
    """
    Computes a weighted average of channels across timesteps (1 parameter pr. channel).
    """
    def __init__(self, attention_size, return_attention=False):
        """ Initialize the attention layer
        # Arguments:
            attention_size: Size of the attention vector.
            return_attention: If true, output will include the weight for each input token
                              used for the prediction
        """
        super(Attention, self).__init__()
        self.return_attention = return_attention
        self.attention_size = attention_size
        self.attention_vector = Parameter(torch.FloatTensor(attention_size))
        torch.nn.init.uniform(self.attention_vector.data, -0.01, 0.01)

    def __repr__(self):
        s = '{name}({attention_size}, return attention={return_attention})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs, input_lengths, return_attention=False):
        """ Forward pass.
        # Arguments:
            inputs (Torch.Variable): Tensor of input sequences
            input_lengths (torch.LongTensor): Lengths of the sequences
        # Return:
            Tuple with (representations and attentions if self.return_attention else None).
        """
        logits = inputs.matmul(self.attention_vector)
        unnorm_ai = (logits - logits.max()).exp()
        # Compute a mask for the attention on the padded sequences
        # See e.g. https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/5
        max_len = unnorm_ai.size(1)

        idxes = torch.arange(
            0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
        idxes = idxes.cuda()
        mask = Variable((idxes < input_lengths.unsqueeze(1)).float())

        # apply mask and renormalize attention scores (weights)
        masked_weights = unnorm_ai * mask
        att_sums = masked_weights.sum(dim=1, keepdim=True)  # sums per sequence
        attentions = masked_weights.div(att_sums)

        # apply attention weights
        weighted = torch.mul(
            inputs, attentions.unsqueeze(-1).expand_as(inputs))
        # get the final fixed vector representations of the sentences
        representations = weighted.sum(dim=1)
        return (representations, attentions if self.return_attention or return_attention else None)
    


class SlotGate(nn.Module):
    def __init__(self, params):
        super(SlotGate, self).__init__()
        self.emb_dim = params.emb_dim
        self.gate_size = params.gate_size  # gate_size should be the same as embedding dimension
        self.w1 = Parameter(torch.FloatTensor(self.gate_size, self.gate_size))
        torch.nn.init.uniform(self.w1.data, -0.01, 0.01)
        self.w2 = Parameter(torch.FloatTensor(self.gate_size, self.gate_size))
        torch.nn.init.uniform(self.w2.data, -0.01, 0.01)
        self.sigmoid = nn.Sigmoid()
        self.ontology_emb_file = params.ontology_emb_file
        self.ontology_emb = load_ontology_embedding(self.ontology_emb_file)
        
    def forward(self, acts_request, acts_slot, acts_value, slot_names):
        '''
        Inputs:
            acts_request: (bsz, var_length)
            acts_slot: (bsz, var_length)
            acts_value: (bsz, var_length)
            slot_names: (bsz, 4) (food, price range, area, request)
        outputs:
            gates: (bsz, 4, gate_size)
        '''
        gates = []
        batch_size = len(acts_request)
        
        for batch_id in range(batch_size):
            t_q = torch.zeros(self.emb_dim).cuda()
            t_s = torch.zeros(self.emb_dim).cuda()
            t_v = torch.zeros(self.emb_dim).cuda()

            # system acts request
            req_list = acts_request[batch_id]
            for req in req_list:
                t_q = t_q + torch.FloatTensor(self.ontology_emb[req]).cuda()
            # system acts slot_type
            slot_type_list = acts_slot[batch_id]
            for slot_type in slot_type_list:
                t_s = t_s + torch.FloatTensor(self.ontology_emb[slot_type]).cuda()
            # system acts slot_value
            slot_value_list = acts_value[batch_id]
            for slot_value in slot_value_list:
                t_v = t_v + torch.FloatTensor(self.ontology_emb[slot_value]).cuda()
            
            slot_name = slot_names[batch_id]
            # slot name: (food, price range, area, request)
            for id_, slot in enumerate(slot_name):
                c_s = torch.FloatTensor(self.ontology_emb[slot]).cuda()
                # gate1
                gate1 = c_s.unsqueeze(0)
                # gate2
                gate2 = self.sigmoid(c_s * torch.matmul(self.w1, t_q)).unsqueeze(0)
                # gate3
                gate3 = self.sigmoid(c_s * torch.matmul(self.w2, (t_s+t_v))).unsqueeze(0)
                
                gate = gate1 + gate2 + gate3
                if id_ == 0:
                    gates_each = gate
                else:
                    gates_each = torch.cat((gates_each, gate), dim=0)
            
            assert gates_each.size() == torch.Size([4, self.emb_dim])
            gates_each = gates_each.unsqueeze(0)
            
            if batch_id == 0:
                gates = gates_each
            else:
                gates = torch.cat((gates, gates_each), dim=0)

        assert gates.size() == torch.Size([batch_size, 4, self.emb_dim])

        return gates

class Predictor(nn.Module):
    def __init__(self, params):
        super(Predictor, self).__init__()
        self.hidden_size = params.hidden_dim * 2 + params.gate_size if params.bidirection else params.hidden_dim + params.gate_size
        self.food_class = params.food_class
        self.price_range_class = params.price_range_class
        self.area_class = params.area_class
        self.request_class = params.request_class

        self.linear_food = nn.Linear(self.hidden_size, self.food_class)
        self.linear_price_range = nn.Linear(self.hidden_size, self.price_range_class)
        self.linear_area = nn.Linear(self.hidden_size, self.area_class)
        self.linear_request = nn.Linear(self.hidden_size, self.request_class)

        self.sigmoid = nn.Sigmoid()

    def forward(self, utter_representation, gates):
        '''
        Inputs:
            utter_representation: (bsz, hidden_dim*2)
            gates: (bsz, 4, gate_size)
        Outputs:
            food_class prediction: (bsz,)
            price_range_class prediction: (bsz,)
            area_class prediction: (bsz,)
            request_class prediction: (bsz,)
        '''
        # food slot
        food_gates = gates[:, 0, :]  # bsz, gate_size
        feature_food = torch.cat((utter_representation, food_gates), dim=1)
        food_value_pred = self.linear_food(feature_food)

        # price range slot
        price_range_gates = gates[:, 1, :]
        feature_price_range = torch.cat((utter_representation, price_range_gates), dim=1)
        price_range_value_pred = self.linear_price_range(feature_price_range)

        # area slot
        area_gates = gates[:, 2, :]
        feature_area = torch.cat((utter_representation, area_gates), dim=1)
        area_value_pred = self.linear_area(feature_area)

        # request
        request_gates = gates[:, 3, :]
        feature_request = torch.cat((utter_representation, request_gates), dim=1)
        request_value_pred = self.sigmoid(self.linear_request(feature_request))

        return food_value_pred, price_range_value_pred, area_value_pred, request_value_pred

class DialogueStateTracker(nn.Module):
    def __init__(self, params, vocab_en, vocab_trans):
        super(DialogueStateTracker, self).__init__()
        # build model
        self.lstm = Lstm(params, vocab_en, vocab_trans)
        self.attention_size = params.hidden_dim * 2 if params.bidirection else params.hidden_dim
        self.atten_layer = Attention(attention_size=self.attention_size, return_attention=False)
        self.slot_gate = SlotGate(params)
        self.predictor = Predictor(params)

    def forward(self, utters, lengths, acts_request, acts_slot, acts_value, slot_name, lang):
        '''
        Inputs:
            utters: (bsz, seq_len)
            acts_request: (bsz, var_length)
            acts_slot: (bsz, var_length)
            acts_value: (bsz, var_length)
            slot_name: (bsz, 4) (food, price range, area, request)
        Outputs:
            food_class prediction: (bsz, 75)
            price_range_class prediction: (bsz, 4)
            area_class prediction: (bsz, 6)
            request_class prediction: (bsz, 7)
        '''
        # utterance representation
        lstm_out = self.lstm(utters, lang)
        utter_repre, _ = self.atten_layer(lstm_out, lengths)
        # gates
        gates = self.slot_gate(acts_request, acts_slot, acts_value, slot_name)
        # predictions
        food_value_pred, price_range_value_pred, area_value_pred, request_value_pred = self.predictor(utter_repre, gates)

        return food_value_pred, price_range_value_pred, area_value_pred, request_value_pred
