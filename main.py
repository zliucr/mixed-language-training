
from config import get_params
from src.utils import init_experiment, load_embedding, load_embedding2
from src.dst_loader import get_dst_dataloader
from src.nlu_loader import get_nlu_dataloader
from src.dst_model import DialogueStateTracker
from src.nlu_model import Lstm_nlu, IntentPredictor, SlotPredictor
from src.trainer import DST_Trainer, NLU_Trainer
from src.preparation import Vocab

import torch
from tqdm import tqdm
import pickle
import json
import codecs
import numpy as np

def train_dst(params):
    # initialize experiment
    logger = init_experiment(params, logger_filename=params.logger_filename)

    with codecs.open(params.ontology_class_path, 'r', 'utf8') as f:
        dialogue_ontology = json.load(f)

    # get vocab and dialogue_ontology
    with open(params.vocab_path_en, "rb") as f:
        vocab_en = pickle.load(f)
    with open(params.vocab_path_trans, "rb") as f:
        vocab_trans = pickle.load(f)

    # dataloader
    dataloader_tr, dataloader_val, dataloader_test = get_dst_dataloader(params, vocab_en, vocab_trans, dialogue_ontology)
    dst_model = DialogueStateTracker(params, vocab_en, vocab_trans)
    dst_model.cuda()

    # build trainer
    dst_trainer = DST_Trainer(params, dst_model)

    for e in range(params.epoch):
        logger.info("============== epoch {} ==============".format(e+1))
        food_loss_list, price_loss_list, area_loss_list, request_loss_list = [], [], [], []

        pbar = tqdm(enumerate(dataloader_tr), total=len(dataloader_tr))
        for i, (_, utters, lengths, acts_request, acts_slot, acts_values, slot_names, turn_slot_labels, turn_request_labels) in pbar:
            turn_slot_labels, turn_request_labels = turn_slot_labels.cuda(), turn_request_labels.cuda()
            utters, lengths = utters.cuda(), lengths.cuda()

            food_loss, price_loss, area_loss, request_loss = dst_trainer.train_step(utters, lengths, acts_request, acts_slot, acts_values, slot_names, turn_slot_labels, turn_request_labels)
            
            food_loss_list.append(food_loss)
            price_loss_list.append(price_loss)
            area_loss_list.append(area_loss)
            request_loss_list.append(request_loss)

            pbar.set_description("(Epoch {}) FOOD:{:.4f} PRICE:{:.4f} AREA:{:.4f} REQUEST:{:.4f}".format(e+1, np.mean(food_loss), np.mean(price_loss), np.mean(area_loss), np.mean(request_loss)))

        logger.info("Finish training epoch {}. FOOD:{:.4f} PRICE:{:.4f} AREA:{:.4f} REQUEST:{:.4f}".format(e+1, np.mean(food_loss), np.mean(price_loss), np.mean(area_loss), np.mean(request_loss)))
        
        logger.info("============== Evaluate {} ==============".format(e+1))
        goal_acc, request_acc, joint_goal_acc, avg_acc, stop_training_flag = dst_trainer.evaluate(dataloader_val, isTestset=False)
        logger.info("({}) Goal ACC: {:.4f}. Joint ACC: {:.4f}. Request ACC: {:.4f}. Avg ACC: {:.4f} (Best Avg Acc: {:.4f})".format(params.trans_lang, goal_acc, joint_goal_acc, request_acc, avg_acc, dst_trainer.best_avg_acc))

        goal_acc, request_acc, joint_goal_acc, avg_acc, _ = dst_trainer.evaluate(dataloader_test, isTestset=True)
        logger.info("({}) Goal ACC: {:.4f}. Joint ACC: {:.4f}. Request ACC: {:.4f}. Avg ACC: {:.4f}".format(params.trans_lang, goal_acc, joint_goal_acc, request_acc, avg_acc))
        
        if stop_training_flag == True:
            break
    
    logger.info("============== Final Test ==============")
    goal_acc, request_acc, joint_goal_acc, avg_acc, _ = dst_trainer.evaluate(dataloader_test, isTestset=True, load_best_model=True)
    logger.info("Goal ACC: {:.4f}. Joint ACC: {:.4f}. Request ACC: {:.4f}. Avg ACC: {:.4f})".format(goal_acc, joint_goal_acc, request_acc, avg_acc))


def train_nlu(params):
    # initialize experiment
    logger = init_experiment(params, logger_filename=params.logger_filename)

    # dataloader
    dataloader_tr, dataloader_val, dataloader_test, vocab_en, vocab_trans = get_nlu_dataloader(params)

    # build model
    lstm = Lstm_nlu(params, vocab_en, vocab_trans)
    
    intent_predictor = IntentPredictor(params)
    slot_predictor = SlotPredictor(params)
    lstm.cuda()
    intent_predictor.cuda()
    slot_predictor.cuda()

    # build trainer
    nlu_trainer = NLU_Trainer(params, lstm, intent_predictor, slot_predictor)

    for e in range(params.epoch):
        logger.info("============== epoch {} ==============".format(e+1))
        intent_loss_list, slot_loss_list = [], []
        
        pbar = tqdm(enumerate(dataloader_tr), total=len(dataloader_tr))
        for i, (X, lengths, y1, y2) in pbar:
            X, lengths, y1 = X.cuda(), lengths.cuda(), y1.cuda()  # the length of y2 is different for each sequence

            intent_loss, slot_loss = nlu_trainer.train_step(X, lengths, y1, y2)
            intent_loss_list.append(intent_loss)
            slot_loss_list.append(slot_loss)
            
            pbar.set_description("(Epoch {}) INTENT LOSS:{:.4f} SLOT LOSS:{:.4f}".format(e+1, np.mean(intent_loss_list), np.mean(slot_loss_list)))
        
        logger.info("Finish training epoch {}. Intent loss: {:.4f}. Slot loss: {:.4f}".format(e+1, np.mean(intent_loss_list), np.mean(slot_loss_list)))
        
        logger.info("============== Evaluate %d ==============" % e)
        intent_acc, slot_f1, stop_training_flag = nlu_trainer.evaluate(dataloader_val)
        logger.info("({}) Intent ACC: {:.4f} (Best Acc: {:.4f}). Slot F1: {:.4f}. (Best F1: {:.4f})".format(params.trans_lang, intent_acc, nlu_trainer.best_intent_acc, slot_f1, nlu_trainer.best_slot_f1))

        intent_acc, slot_f1, _ = nlu_trainer.evaluate(dataloader_test, istestset=True)
        logger.info("({}) Intent ACC: {:.4f}. Slot F1: {:.4f}.".format(params.trans_lang, intent_acc, slot_f1))
        
        if stop_training_flag == True:
            break
    
    logger.info("============== Final Test ==============")
    intent_acc, slot_f1, _ = nlu_trainer.evaluate(dataloader_test, istestset=True, load_best_model=True)
    logger.info("Intent ACC: {:.4f}. Slot F1: {:.4f}.".format(intent_acc, slot_f1))

if __name__ == "__main__":
    params = get_params()

    if params.run_nlu:
        train_nlu(params)
    else:
        train_dst(params)
