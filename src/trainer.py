
import torch
import torch.nn as nn

from tqdm import tqdm
import numpy as np
import os
import logging
logger = logging.getLogger()

from sklearn.metrics import accuracy_score, f1_score
from src.conll2002_metrics import *

class DST_Trainer(object):
    def __init__(self, params, dst_model):
        self.dst_model = dst_model
        self.lr = params.lr
        self.params = params

        # Adam optimizer
        self.optimizer = torch.optim.Adam(dst_model.parameters(), lr=self.lr, weight_decay=params.weight_decay)
        self.loss_fn1 = nn.CrossEntropyLoss()
        self.loss_fn2 = nn.MSELoss()

        self.early_stop = params.early_stop
        self.no_improvement_num = 0
        self.stop_training_flag = False
        self.best_avg_acc = 0
        # self.best_goal_acc = 0

    def train_step(self, utters, lengths, acts_request, acts_slot, acts_value, slot_name, slot_labels, request_labels):
        self.dst_model.train()

        food_value_pred, price_range_value_pred, area_value_pred, request_value_pred = self.dst_model(utters, lengths, acts_request, acts_slot, acts_value, slot_name, "en")

        # slot value labels
        # slot_labels: (bsz, 3)
        food_label = slot_labels[:, 0]  # (bsz, 1)
        price_range_label = slot_labels[:, 1] # (bsz, 1)
        area_label = slot_labels[:, 2] # (bsz, 1)
        self.optimizer.zero_grad()
        food_pred_loss = self.loss_fn1(food_value_pred, food_label)
        food_pred_loss.backward(retain_graph=True)
        self.optimizer.step()
        
        self.optimizer.zero_grad()
        price_range_pred_loss = self.loss_fn1(price_range_value_pred, price_range_label)
        price_range_pred_loss.backward(retain_graph=True)
        self.optimizer.step()

        self.optimizer.zero_grad()
        area_pred_loss = self.loss_fn1(area_value_pred, area_label)
        area_pred_loss.backward(retain_graph=True)
        self.optimizer.step()

        # request label
        # request_labels: (bsz, 7)
        self.optimizer.zero_grad()
        request_pred_loss = self.loss_fn2(request_value_pred, request_labels)
        request_pred_loss.backward(retain_graph=True)
        self.optimizer.step()

        return food_pred_loss.item(), price_range_pred_loss.item(), area_pred_loss.item(), request_pred_loss.item()
    
    def evaluate(self, dataloader, isTestset=False, load_best_model=False):
        if load_best_model == True:
            # load best model
            best_model_path = os.path.join(self.params.dump_path, "best_model.pth")
            logger.info("Loading best model from %s" % best_model_path)
            best_model = torch.load(best_model_path)
            self.dst_model = best_model["dialog_state_tracker"]

        self.dst_model.eval()

        # collect predictions and labels
        y_food, y_price, y_area, y_request = [], [], [], []
        pred_food, pred_price, pred_area, pred_request = [], [], [], []
        dialogue_indices = []
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (dialgue_idx, utters, lengths, acts_request, acts_slot, acts_values, slot_names, turn_slot_labels, turn_request_labels) in pbar:
            # slot labels
            turn_slot_labels = turn_slot_labels.data.cpu().numpy()
            y_food.append(turn_slot_labels[:, 0])
            y_price.append(turn_slot_labels[:, 1])
            y_area.append(turn_slot_labels[:, 2])
            # request labels
            turn_request_labels = turn_request_labels.data.cpu().numpy()
            y_request.append(turn_request_labels)
            utters, lengths = utters.cuda(), lengths.cuda()

            food_value_pred, price_range_value_pred, area_value_pred, request_value_pred = self.dst_model(utters, lengths, acts_request, acts_slot, acts_values, slot_names, self.params.trans_lang)
            
            # slot value prediction
            dialogue_indices.extend(dialgue_idx)
            pred_food.append(food_value_pred.detach().data.cpu().numpy())
            pred_price.append(price_range_value_pred.detach().data.cpu().numpy())
            pred_area.append(area_value_pred.detach().data.cpu().numpy())
            pred_request.append(request_value_pred.detach().data.cpu().numpy())

        # evaluate
        y_food = np.concatenate(y_food, axis=0)
        y_price = np.concatenate(y_price, axis=0)
        y_area = np.concatenate(y_area, axis=0)
        y_request = np.concatenate(y_request, axis=0)
        pred_food = np.concatenate(pred_food, axis=0)
        pred_food = np.argmax(pred_food, axis=1)
        pred_price = np.concatenate(pred_price, axis=0)
        pred_price = np.argmax(pred_price, axis=1)
        pred_area = np.concatenate(pred_area, axis=0)
        pred_area = np.argmax(pred_area, axis=1)

        pred_request = np.concatenate(pred_request, axis=0)
        pred_request = (pred_request > 0.5) * 1.0

        assert len(y_food) == len(y_price) == len(y_area) == len(y_request) == len(pred_food) == len(pred_price) == len(pred_area) == len(pred_request) == len(dialogue_indices)

        joint_goal_total, joint_goal_correct = 0, 0
        goal_total, goal_correct = 0, 0
        request_total, request_correct = 0, 0
        for i in range(len(y_food)):
            y_food_ = y_food[i]
            y_price_ = y_price[i]
            y_area_ = y_area[i]

            dialog_idx = dialogue_indices[i]
            pred_food_ = pred_food[i]
            pred_price_ = pred_price[i]
            pred_area_ = pred_area[i]

            if i == 0: assert dialog_idx == 0
            
            if dialog_idx != 0:
                if pre_pred_food_ != self.params.food_class-1 and pred_food_ == self.params.food_class-1:
                    pred_food_ = pre_pred_food_

                if pre_pred_price_ != self.params.price_range_class-1 and pred_price_ == self.params.price_range_class-1:
                    pred_price_ = pre_pred_price_

                if pre_pred_area_ != self.params.area_class-1 and pred_area_ == self.params.area_class-1:
                    pred_area_ = pre_pred_area_

            joint_goal_total += 1
            if y_food_ == pred_food_ and y_price_ == pred_price_ and y_area_ == pred_area_:
                joint_goal_correct += 1
            
            goal_total += 1
            if y_food_ == pred_food_:
                goal_correct += 1
            
            goal_total += 1
            if y_price_ == pred_price_:
                goal_correct += 1
            
            goal_total += 1
            if y_area_ == pred_area_:
                goal_correct += 1

            pre_pred_food_ = pred_food_
            pre_pred_price_ = pred_price_
            pre_pred_area_ = pred_area_

            y_request_ = y_request[i]
            pred_request_ = pred_request[i]
            request_total += 1
            if np.array_equal(y_request_, pred_request_) == True:
                request_correct += 1
        
        joint_goal_acc = joint_goal_correct * 1.0 / joint_goal_total
        goal_acc = goal_correct * 1.0 / goal_total
        request_acc = request_correct * 1.0 / request_total
        avg_acc = (joint_goal_acc + request_acc) / 2

        if isTestset == False:
            if avg_acc > self.best_avg_acc:
                self.best_avg_acc = avg_acc
                self.no_improvement_num = 0
                self.save_model()
            else:
                self.no_improvement_num += 1
                logger.info("No better model found (%d/%d)" % (self.no_improvement_num, self.early_stop))
            
        if self.no_improvement_num >= self.early_stop:
            self.stop_training_flag = True
        
        return goal_acc, request_acc, joint_goal_acc, avg_acc, self.stop_training_flag
        
    def save_model(self):
        """
        save the best model (achieve best f1 on slot prediction)
        """
        saved_path = os.path.join(self.params.dump_path, "best_model.pth")
        torch.save({
            "dialog_state_tracker": self.dst_model
        }, saved_path)
        
        logger.info("Best model has been saved to %s" % saved_path)


index2slot = ['O', 'B-weather/noun', 'I-weather/noun', 'B-location', 'I-location', 'B-datetime', 'I-datetime', 'B-weather/attribute', 'I-weather/attribute', 'B-reminder/todo', 'I-reminder/todo', 'B-alarm/alarm_modifier', 'B-reminder/noun', 'B-reminder/recurring_period', 'I-reminder/recurring_period', 'B-reminder/reference', 'I-reminder/noun', 'B-reminder/reminder_modifier', 'I-reminder/reference', 'I-reminder/reminder_modifier', 'B-weather/temperatureUnit', 'I-alarm/alarm_modifier', 'B-alarm/recurring_period', 'I-alarm/recurring_period']

class NLU_Trainer(object):
    def __init__(self, params, lstm, intent_predictor, slot_predictor):
        self.lstm = lstm
        self.intent_predictor = intent_predictor
        self.slot_predictor = slot_predictor
        self.lr = params.lr
        self.params = params
        
        model = [
            {"params": self.lstm.parameters()},
            {"params": self.intent_predictor.parameters()},
            {"params": self.slot_predictor.parameters()}
        ]
        
        self.optimizer = torch.optim.Adam(model, lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()

        self.early_stop = params.early_stop
        self.no_improvement_num = 0
        self.best_intent_acc = 0
        self.best_slot_f1 = 0
        
        self.stop_training_flag = False

    def train_step(self, X, lengths, y1, y2):
        self.lstm.train()
        self.intent_predictor.train()
        self.slot_predictor.train()

        lstm_layer = self.lstm(X, "en")
        intent_prediction = self.intent_predictor(lstm_layer, lengths)
        
        # train IntentPredictor
        self.optimizer.zero_grad()
        intent_loss = self.loss_fn(intent_prediction, y1)
        intent_loss.backward(retain_graph=True)
        self.optimizer.step()

        # train SlotPredictor
        slot_prediction = self.slot_predictor(lstm_layer)
        slot_loss = self.slot_predictor.crf_loss(slot_prediction, lengths, y2)
        self.optimizer.zero_grad()
        slot_loss.backward()
        self.optimizer.step()

        return intent_loss.item(), slot_loss.item()

    def evaluate(self, dataloader, istestset=False, load_best_model=False):
        if load_best_model == True:
            # load best model
            best_model_path = os.path.join(self.params.dump_path, "best_model.pth")
            logger.info("Loading best model from %s" % best_model_path)
            best_model = torch.load(best_model_path)
            self.lstm = best_model["text_encoder"]
            self.intent_predictor = best_model["intent_predictor"]
            self.slot_predictor = best_model["slot_predictor"]
        
        self.lstm.eval()
        self.intent_predictor.eval()
        self.slot_predictor.eval()
        intent_pred, slot_pred = [], []
        y1_list, y2_list = [], []
        pbar = tqdm(enumerate(dataloader),total=len(dataloader))
        for i, (X, lengths, y1, y2) in pbar:
            y1_list.append(y1.data.cpu().numpy())
            y2_list.extend(y2) # y2 is a list
            X, lengths = X.cuda(), lengths.cuda()

            lstm_layer = self.lstm(X, self.params.trans_lang)
            intent_prediction = self.intent_predictor(lstm_layer, lengths)
            # for intent_pred
            intent_pred.append(intent_prediction.data.cpu().numpy())
            # for slot_pred
            slot_prediction = self.slot_predictor(lstm_layer)
            slot_pred_batch = self.slot_predictor.crf_decode(slot_prediction, lengths)
            slot_pred.extend(slot_pred_batch)

        # concatenation
        intent_pred = np.concatenate(intent_pred, axis=0)
        intent_pred = np.argmax(intent_pred, axis=1)
        slot_pred = np.concatenate(slot_pred, axis=0)
        y1_list = np.concatenate(y1_list, axis=0)
        y2_list = np.concatenate(y2_list, axis=0)
        intent_acc = accuracy_score(y1_list, intent_pred)

        # calcuate f1 score
        # slot_f1 = f1_score(y2_list, slot_pred, average="macro")
        y2_list = list(y2_list)
        slot_pred = list(slot_pred)
        lines = []
        for pred_index, gold_index in zip(slot_pred, y2_list):
            pred_slot = index2slot[pred_index]
            gold_slot = index2slot[gold_index]
            lines.append("w" + " " + pred_slot + " " + gold_slot)
        results = conll2002_measure(lines)
        slot_f1 = results["fb1"]
        
        if istestset == False:
            if intent_acc > self.best_intent_acc:
                self.best_intent_acc = intent_acc
            if slot_f1 > self.best_slot_f1:
                self.best_slot_f1 = slot_f1
                self.no_improvement_num = 0
                # only when best slot_f1 is found, we save the model
                self.save_model()
            else:
                self.no_improvement_num += 1
                logger.info("No better model found (%d/%d)" % (self.no_improvement_num, self.early_stop))
        
        if self.no_improvement_num >= self.early_stop:
            self.stop_training_flag = True

        return intent_acc, slot_f1, self.stop_training_flag

    def save_model(self):
        """
        save the best model (achieve best f1 on slot prediction)
        """
        saved_path = os.path.join(self.params.dump_path, "best_model.pth")
        torch.save({
            "text_encoder": self.lstm,
            "intent_predictor": self.intent_predictor,
            "slot_predictor": self.slot_predictor
        }, saved_path)
        
        logger.info("Best model has been saved to %s" % saved_path)
