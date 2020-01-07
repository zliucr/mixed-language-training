
from src.preparation import Vocab
from src.utils import binarize_nlu_data

from copy import deepcopy
import codecs
import json
import csv
import re
import string
import os
import pickle

import logging
logger = logging.getLogger()

def load_woz_data(file_path, language, dialogue_ontology, mapping=None):
    """
    This method loads WOZ dataset as a collection of utterances.

    Testing means load everything, no split.
    """
    with codecs.open(file_path, 'r', 'utf8') as f:
        woz_json = json.load(f)

    turns = []
    dialogue_count = len(woz_json)

    logger.info("loading from file {} totally {} dialogues".format(file_path, dialogue_count))
    
    for idx in range(0, dialogue_count):
        current_dialogue = process_woz_dialogue(woz_json[idx]["dialogue"], language, dialogue_ontology, mapping=mapping)
        turns.extend(current_dialogue)
    
    return turns

def process_woz_dialogue(woz_dialogue, language, dialogue_ontology, mapping=None):
    """
    Returns a list of (tuple, belief_state) for each turn in the dialogue.
    """
    # initial belief state
    # belief state to be given at each turn
    if language == "english" or language == "en":
        null_bs = {}
        null_bs["food"] = "none"
        null_bs["price range"] = "none"
        null_bs["area"] = "none"
        null_bs["request"] = []
        informable_slots = ["food", "price range", "area"]
        pure_requestables = ["address", "phone", "postcode"]

    elif (language == "italian" or language == "it"):
        null_bs = {}
        null_bs["cibo"] = "none"
        null_bs["prezzo"] = "none"
        null_bs["area"] = "none"
        null_bs["request"] = []
        informable_slots = ["cibo", "prezzo", "area"]
        pure_requestables = ["codice postale", "telefono", "indirizzo"]

    elif (language == "german" or language == "de"):
        null_bs = {}
        null_bs["essen"] = "none"
        null_bs["preisklasse"] = "none"
        null_bs["gegend"] = "none"
        null_bs["request"] = []
        informable_slots = ["essen", "preisklasse", "gegend"]
        pure_requestables = ["postleitzahl", "telefon", "adresse"]
    else:
        null_bs = {}
        pure_requestables = None

    prev_belief_state = deepcopy(null_bs)
    dialogue_representation = []

    for idx, turn in enumerate(woz_dialogue):

        current_DA = turn["system_acts"]

        current_req = []
        current_conf_slot = []
        current_conf_value = []

        for each_da in current_DA:
            if each_da in informable_slots:
                current_req.append(each_da)
            elif each_da in pure_requestables:
                current_conf_slot.append("request")
                current_conf_value.append(each_da)
            else:
                if type(each_da) is list:
                    current_conf_slot.append(each_da[0])
                    current_conf_value.append(each_da[1])

        current_transcription = turn["transcript"]

        # exclude = set(string.punctuation)
        # exclude.remove("'")

        # current_transcription = ''.join(ch for ch in current_transcription if ch not in exclude)
        if mapping == None or language != "en":
            current_transcription = current_transcription.lower()
        else:
            for key, value in mapping.items():
                if len(key.split()) > 1:
                    if key == "price range":  ## could be price ranges in the utterance
                        current_transcription = current_transcription.replace("price ranges", value)
                    current_transcription = current_transcription.replace(key, value)
                else:
                    splits = current_transcription.split()
                    for i, word in enumerate(splits):
                        if word == key: splits[i] = value
                    current_transcription = " ".join(splits)

        current_labels = turn["turn_label"]

        turn_bs = deepcopy(null_bs)
        current_bs = deepcopy(prev_belief_state)

        # print "=====", prev_belief_state
        if "request" in prev_belief_state:
            del prev_belief_state["request"]

        current_bs["request"] = []  # reset requestables at each turn
        
        legal_flag = True
        for label in current_labels:
            (c_slot, c_value) = label
            c_value = c_value.strip()
            
            # remove those illegal slot value
            if language == "en" and (c_value not in dialogue_ontology[c_slot]["en"]):
                legal_flag = False
                break

            if c_slot in informable_slots:
                current_bs[c_slot] = c_value
                turn_bs[c_slot] = c_value
            elif c_slot == "request":
                current_bs["request"].append(c_value)
                turn_bs["request"].append(c_value)

        if legal_flag == True:
            dialogue_representation.append((idx, current_transcription, current_req, current_conf_slot, current_conf_value, deepcopy(current_bs), deepcopy(turn_bs)))

            prev_belief_state = deepcopy(current_bs)

    return dialogue_representation


# for dialogue NLU dataset
def get_vocab(word_set):
    vocab = Vocab()
    vocab.index_words(word_set)
    return vocab

# for dialogue NLU dataset
def parse_tsv(data_path, intent_set=[], slot_set=["O"], istrain=True):
    """
    Input: 
        data_path: the path of data
        intent_set: set of intent (empty if it is train data)
        slot_set: set of slot type (empty if it is train data)
    Output:
        data_tsv: {"text": [[token1, token2, ...], ...], "slot": [[slot_type1, slot_type2, ...], ...], "intent": [intent_type, ...]}
        intent_set: set of intent
        slot_set: set of slot type
    """
    slot_type_list = ["alarm", "datetime", "location", "reminder", "weather"]
    data_tsv = {"text": [], "slot": [], "intent": []}
    with open(data_path) as tsv_file:
        reader = csv.reader(tsv_file, delimiter="\t")
        for i, line in enumerate(reader):
            intent = line[0]
            if istrain == True and intent not in intent_set: intent_set.append(intent)
            if istrain == False and intent not in intent_set:
                intent_set.append(intent)
                # logger.info("Found intent %s not in train data" % intent)
                # print("Found intent %s not in train data" % intent)
            slot_splits = line[1].split(",")
            slot_line = []
            slot_flag = True
            if line[1] != '':
                for item in slot_splits:
                    item_splits = item.split(":")
                    assert len(item_splits) == 3
                    # slot_item = {"start": item_splits[0], "end": item_splits[1], "slot": item_splits[2].split("/")[0]}
                    slot_item = {"start": item_splits[0], "end": item_splits[1], "slot": item_splits[2]}
                    flag = False
                    for slot_type in slot_type_list:
                        if slot_type in slot_item["slot"]:
                            flag = True

                    if flag == False:
                        slot_flag = False
                        break
                    # if istrain == True and slot_item["slot"] not in slot_set: slot_set.append(slot_item["slot"])
                    # if istrain == False and slot_item["slot"] not in slot_set:
                    #     slot_set.append(slot_item["slot"])
                    #     # logger.info("Found slot %s not in train data" % item_splits[2])
                    #     # print("Found slot %s not in train data" % item_splits[2])
                    slot_line.append(slot_item)
            
            if slot_flag == False:
                # slot flag not correct
                continue

            token_part = json.loads(line[4])
            tokens = token_part["tokenizations"][0]["tokens"]
            tokenSpans = token_part["tokenizations"][0]["tokenSpans"]

            data_tsv["text"].append(tokens)
            data_tsv["intent"].append(intent)
            slots = []
            for tokenspan in tokenSpans:
                nolabel = True
                for slot_item in slot_line:
                    start = tokenspan["start"]
                    # if int(start) >= int(slot_item["start"]) and int(start) < int(slot_item["end"]):
                    if int(start) == int(slot_item["start"]):
                        nolabel = False
                        slot_ = "B-" + slot_item["slot"]
                        slots.append(slot_)
                        if slot_ not in slot_set:
                            slot_set.append(slot_)
                        break
                    if int(start) > int(slot_item["start"]) and int(start) < int(slot_item["end"]):
                        nolabel = False
                        slot_ = "I-" + slot_item["slot"]
                        slots.append(slot_)
                        if slot_ not in slot_set:
                            slot_set.append(slot_)
                        break
                if nolabel == True: slots.append("O")
            data_tsv["slot"].append(slots)

            assert len(slots) == len(tokens)

    return data_tsv, intent_set, slot_set

# for dialogue NLU dataset
def clean_text(data, lang):
    # detect pattern
    # detect <TIME>
    pattern_time1 = re.compile(r"[0-9]+[ap]")
    pattern_time2 = re.compile(r"[0-9]+[;.h][0-9]+")
    pattern_time3 = re.compile(r"[ap][.][am]")
    pattern_time4 = range(2000, 2020)
    # pattern_time5: token.isdigit() and len(token) == 3

    pattern_time_th1 = re.compile(r"[\u0E00-\u0E7F]+[0-9]+")
    pattern_time_th2 = re.compile(r"[0-9]+[.]*[0-9]*[\u0E00-\u0E7F]+")
    pattern_time_th3 = re.compile(r"[0-9]+[.][0-9]+")

    # detect <LAST>
    pattern_last1 = re.compile(r"[0-9]+min")
    pattern_last2 = re.compile(r"[0-9]+h")
    pattern_last3 = re.compile(r"[0-9]+sec")

    # detect <DATE>
    pattern_date1 = re.compile(r"[0-9]+st")
    pattern_date2 = re.compile(r"[0-9]+nd")
    pattern_date3 = re.compile(r"[0-9]+rd")
    pattern_date4 = re.compile(r"[0-9]+th")

    # detect <LOCATION>: token.isdigit() and len(token) == 5
    
    # detect <NUMBER>: token.isdigit()
    
    # for English: replace contain n't with not
    # for English: remove 's, 'll, 've, 'd, 'm
    remove_list = ["'s", "'ll", "'ve", "'d", "'m"]
    
    data_clean = {"text": [], "slot": [], "intent": []}
    data_clean["slot"] = data["slot"]
    data_clean["intent"] = data["intent"]
    for token_list in data["text"]:
        token_list_clean = []
        for token in token_list:
            new_token = token
            # detect <TIME>
            if lang != "th" and ( bool(re.match(pattern_time1, token)) or bool(re.match(pattern_time2, token)) or bool(re.match(pattern_time3, token)) or token in pattern_time4 or (token.isdigit() and len(token)==3) ):
                new_token = "<TIME>"
                token_list_clean.append(new_token)
                continue
            if lang == "th" and ( bool(re.match(pattern_time_th1, token)) or bool(re.match(pattern_time_th2, token)) or bool(re.match(pattern_time_th3, token)) ):
                new_token = "<TIME>"
                token_list_clean.append(new_token)
                continue
            # detect <LAST>
            if lang == "en" and ( bool(re.match(pattern_last1, token)) or bool(re.match(pattern_last2, token)) or bool(re.match(pattern_last3, token)) ):
                new_token = "<LAST>"
                token_list_clean.append(new_token)
                continue
            # detect <DATE>
            if lang == "en" and ( bool(re.match(pattern_date1, token)) or bool(re.match(pattern_date2, token)) or bool(re.match(pattern_date3, token)) or bool(re.match(pattern_date4, token)) ):
                new_token = "<DATE>"
                token_list_clean.append(new_token)
                continue
            # detect <LOCATION>
            if lang != "th" and ( token.isdigit() and len(token)==5 ):
                new_token = "<LOCATION>"
                token_list_clean.append(new_token)
                continue
            # detect <NUMBER>
            if token.isdigit():
                new_token = "<NUMBER>"
                token_list_clean.append(new_token)
                continue
            if lang == "en" and ("n't" in token):
                new_token = "not"
                token_list_clean.append(new_token)
                continue
            if lang == "en":
                for item in remove_list:
                    if item in token:
                        new_token = token.replace(item, "")
                        break

            token_list_clean.append(new_token)
        
        assert len(token_list_clean) == len(token_list)
        data_clean["text"].append(token_list_clean)
    
    return data_clean

def gen_mix_lang_data(data, token_mapping):
    data_new = {"text": [], "slot": [], "intent": []}
    data_new["slot"] = data["slot"]
    data_new["intent"] = data["intent"]
    for token_list in data["text"]:
        token_list_new = []
        for token in token_list:
            if token in token_mapping:
                token = token_mapping[token]
            token_list_new.append(token)
        
        assert len(token_list_new) == len(token_list)
        data_new["text"].append(token_list_new)
    
    return data_new

# for dialogue NLU dataset
def preprocess_nlu_data(data, lang, clean_txt=True, token_mapping=None, vocab_path=None, filtered=False, filtered_scale=None):
    # preprocess from raw (lang) data
    # print("============ Preprocess %s data ============" % lang)
    logger.info("============ Preprocess %s data ============" % lang)

    data_folder = os.path.join('./nlu_data/', lang)
    train_path = os.path.join(data_folder, "train-%s.tsv" % lang)
    eval_path = os.path.join(data_folder, "eval-%s.tsv" % lang)
    # test_path = os.path.join(data_folder, "test-%s.tsv" % lang)
    if lang != "en" and filtered == True:
        print("testing filtering data")
        test_path = os.path.join(data_folder, "test-%s.filter.%s.tsv" % (lang, filtered_scale))
    else:
        test_path = os.path.join(data_folder, "test-%s.tsv" % lang)

    data_train, intent_set, slot_set = parse_tsv(train_path)
    data_eval, intent_set, slot_set = parse_tsv(eval_path, intent_set=intent_set, slot_set=slot_set, istrain=False)
    data_test, intent_set, slot_set = parse_tsv(test_path, intent_set=intent_set, slot_set=slot_set, istrain=False)

    assert len(intent_set) == len(set(intent_set))
    assert len(slot_set) == len(set(slot_set))

    # logger.info("number of intent in %s is %s" % (lang, len(intent_set)))
    # logger.info("number of slot in %s is %s" % (lang, len(slot_set)))
    # print("number of intent in %s is %s" % (lang, len(intent_set)))
    # print("number of slot in %s is %s" % (lang, len(slot_set)))
    
    if lang == "en" and token_mapping is not None:
        logger.info("generating mixed language training data")
        data_train = gen_mix_lang_data(data_train, token_mapping)
        data_eval = gen_mix_lang_data(data_eval, token_mapping)
        data_eval = gen_mix_lang_data(data_eval, token_mapping)

    if clean_txt == True:
        # clean_data
        logger.info("cleaning data on %s language" % lang)
        data_train = clean_text(data_train, lang)
        data_eval = clean_text(data_eval, lang)
        data_test = clean_text(data_test, lang)

    assert vocab_path is not None
    logger.info("Loading vocab from %s" % vocab_path)
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    # logger.info("vocab size of %s is %d" % (lang, vocab.word_num))
    # print("vocab size of %s is %d" % (lang, vocab.word_num))

    data_train_bin = binarize_nlu_data(data_train, intent_set, slot_set, vocab)
    data_eval_bin = binarize_nlu_data(data_eval, intent_set, slot_set, vocab)
    data_test_bin = binarize_nlu_data(data_test, intent_set, slot_set, vocab)
    data[lang] = {"train": data_train_bin, "eval": data_eval_bin, "test": data_test_bin, "vocab": vocab}

