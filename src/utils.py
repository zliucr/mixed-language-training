import os
import subprocess
import pickle
import logging
import time
import random
import codecs
import json
import torch

from datetime import timedelta
import numpy as np
from tqdm import tqdm

import logging
logger = logging.getLogger()

def init_experiment(params, logger_filename):
    """
    Initialize the experiment:
    - save parameters
    - create a logger
    """
    # save parameters
    get_saved_path(params)
    pickle.dump(params, open(os.path.join(params.dump_path, "params.pkl"), "wb"))

    # create a logger
    logger = create_logger(os.path.join(params.dump_path, logger_filename))
    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v))
                          for k, v in sorted(dict(vars(params)).items())))
    logger.info('The experiment will be stored in %s\n' % params.dump_path)

    return logger

class LogFormatter():

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''

def create_logger(filepath):
    # create log formatter
    log_formatter = LogFormatter()
    
    # create file handler and set level to debug
    if filepath is not None:
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    return logger

def get_saved_path(params):
    """
    create a directory to store the experiment
    """
    dump_path = "./" if params.dump_path == "" else params.dump_path
    if not os.path.isdir(dump_path):
        subprocess.Popen("mkdir -p %s" % dump_path, shell=True).wait()
    assert os.path.isdir(dump_path)

    # create experiment path if it does not exist
    exp_path = os.path.join(dump_path, params.exp_name)
    if not os.path.exists(exp_path):
        subprocess.Popen("mkdir -p %s" % exp_path, shell=True).wait()
    
    # generate id for this experiment
    if params.exp_id == "":
        chars = "0123456789"
        while True:
            exp_id = "".join(random.choice(chars) for _ in range(0, 3))
            if not os.path.isdir(os.path.join(exp_path, exp_id)):
                break
    else:
        exp_id = params.exp_id
    # update dump_path
    params.dump_path = os.path.join(exp_path, exp_id)
    if not os.path.isdir(params.dump_path):
        subprocess.Popen("mkdir -p %s" % params.dump_path, shell=True).wait()
    assert os.path.isdir(params.dump_path)

def binarize_dst_data(params, turns, vocab, dialogue_ontology, lang, isTestset=False):
    if lang == "en":
        class_type_dict = {"food": [], "price range": [], "area": [], "request": []}
    elif lang == "de":
        class_type_dict = {"essen": [], "preisklasse": [], "gegend": [], "request": []}
        de2en_mapping = {"essen": "food", "preisklasse": "price range", "gegend": "area", "request": "request"}
    elif lang == "it":
        class_type_dict = {"cibo": [], "prezzo": [], "area": [], "request": []}
        it2en_mapping = {"cibo": "food", "prezzo": "price range", "area": "area", "request": "request"}
    
    for slot_type in class_type_dict.keys():
        if lang == "de":
            slot_type_ = de2en_mapping[slot_type]
        elif lang == "it":
            slot_type_ = it2en_mapping[slot_type]
        else:
            slot_type_ = slot_type
        class_type_dict[slot_type] = dialogue_ontology[slot_type_][lang]

    with codecs.open(params.ontology_mapping_path, 'r', 'utf8') as f:
        ontology_mapping = json.load(f)
    lang_dict = {"en":0, "de":1, "it":2}
    ontology_vocab = []
    lang_id = lang_dict[lang]
    for item in ontology_mapping:
        ontology_vocab.append(item[lang_id])

    binarized_turns = []

    logger.info("Binarizing data ...")
    for i, each_turn in enumerate(turns):
        binarized_utter, binarized_acts_inform_slot, binarized_acts_slot_type, binarized_acts_slot_value, binarized_slots, binarized_slot_values, binarized_request_values = [], [], [], [], [], [], []
        
        dialogue_idx, utterance, acts_inform_slots, acts_slot_type, acts_slot_value, test_label, turn_label = each_turn
        
        labels = test_label if isTestset == True else turn_label
        
        # binarize data
        utter_splits = utterance.split()
        binarized_utter = [ vocab.word2index[tok] for tok in utter_splits ]
        binarized_acts_inform_slot = [ ontology_vocab.index(slot) for slot in acts_inform_slots ]
        binarized_acts_slot_type = [ ontology_vocab.index(s_type) for s_type in acts_slot_type ]
        binarized_acts_slot_value = [ ontology_vocab.index(s_value) for s_value in acts_slot_value ]
        
        for label_slot, label_value in labels.items():

            binarized_slots.append(ontology_vocab.index(label_slot))
            if label_value == "none":
                binarized_slot_values.append(len(class_type_dict[label_slot])) # label for none
            else:
                if type(label_value) is list:
                    r_label = [0] * params.request_class
                    r_indices = [ class_type_dict[label_slot].index(item) for item in label_value ]
                    for idx in r_indices:
                        r_label[idx] = 1
                    binarized_request_values = r_label
                else:
                    binarized_slot_values.append(class_type_dict[label_slot].index(label_value))
        
        current_turn = (dialogue_idx, binarized_utter, binarized_acts_inform_slot, binarized_acts_slot_type, binarized_acts_slot_value, binarized_slots, binarized_slot_values, binarized_request_values)
        
        binarized_turns.append(current_turn)
    
    return binarized_turns


# for dialogue NLU dataset
intent_set = ['weather/find', 'alarm/set_alarm', 'alarm/show_alarms', 'reminder/set_reminder', 'alarm/modify_alarm', 'weather/checkSunrise', 'weather/checkSunset', 'alarm/snooze_alarm', 'alarm/cancel_alarm', 'reminder/show_reminders', 'reminder/cancel_reminder', 'alarm/time_left_on_alarm']
slot_set = ['O', 'B-weather/noun', 'I-weather/noun', 'B-location', 'I-location', 'B-datetime', 'I-datetime', 'B-weather/attribute', 'I-weather/attribute', 'B-reminder/todo', 'I-reminder/todo', 'B-alarm/alarm_modifier', 'B-reminder/noun', 'B-reminder/recurring_period', 'I-reminder/recurring_period', 'B-reminder/reference', 'I-reminder/noun', 'B-reminder/reminder_modifier', 'I-reminder/reference', 'I-reminder/reminder_modifier', 'B-weather/temperatureUnit', 'I-alarm/alarm_modifier', 'B-alarm/recurring_period', 'I-alarm/recurring_period']

def binarize_nlu_data(data, intent_set, slot_set, vocab):
    data_bin = {"text": [], "slot": [], "intent": []}
    # binarize intent
    for intent in data["intent"]:
        index = intent_set.index(intent)
        data_bin["intent"].append(index)
    # binarize text
    for text_tokens in data["text"]:
        text_bin = []
        for token in text_tokens:
            text_bin.append(vocab.word2index[token])
        data_bin["text"].append(text_bin)
    # binarize slot
    for slot in data["slot"]:
        slot_bin = []
        for slot_item in slot:
            index = slot_set.index(slot_item)
            slot_bin.append(index)
        data_bin["slot"].append(slot_bin)
    
    assert len(data_bin["slot"]) == len(data_bin["text"]) == len(data_bin["intent"])
    for text, slot in zip(data_bin["text"], data_bin["slot"]):
        assert len(text) == len(slot)

    return data_bin

def load_embedding(emb_file):
    logger = logging.getLogger()
    logger.info('Loading embedding file: %s' % emb_file)

    embedding = np.load(emb_file)

    return embedding

def load_embedding2(vocab, emb_dim, emb_file):
    logger = logging.getLogger()
    embedding = np.zeros((vocab.n_words, emb_dim))
    logger.info("embedding: %d x %d" % (vocab.n_words, emb_dim))
    assert emb_file is not None
    with open(emb_file, "r") as ef:
        logger.info('Loading embedding file: %s' % emb_file)
        pre_trained = 0
        embedded_words = []
        for i, line in enumerate(ef):
            if i == 0: continue # first line would be "num of words and dimention"
            line = line.strip()
            sp = line.split()
            try:
                assert len(sp) == emb_dim + 1
            except:
                continue
            if sp[0] in vocab.word2index and sp[0] not in embedded_words:
                pre_trained += 1
                embedding[vocab.word2index[sp[0]]] = [float(x) for x in sp[1:]]
                embedded_words.append(sp[0])
        logger.info("Pre-train: %d / %d (%.2f)" % (pre_trained, vocab.n_words, pre_trained / vocab.n_words))

    return embedding

def load_ontology_embedding(ontology_emb_file):
    logger = logging.getLogger()
    logger.info('Loading ontology embedding file: %s' % ontology_emb_file)
    
    ontology_embeddings = np.load(ontology_emb_file)

    return ontology_embeddings
