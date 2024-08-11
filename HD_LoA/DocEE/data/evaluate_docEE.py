import os
import json
import copy
from itertools import chain
import csv
import string
import re
import logging
from collections import defaultdict
import random
random.seed(0)

logger = logging.getLogger(__name__)


def eval_rpf(gt_num, pred_num, correct_num):
    recall = correct_num / gt_num if gt_num != 0 else .0
    precision = correct_num / pred_num if pred_num != 0 else .0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 1e-4 else .0
    res = {
        "recall": recall, "precision": precision, "f1": f1,
        "gt_num": gt_num, "pred_num": pred_num, "correct_num": correct_num,
    }
    return res


def _normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace. (Squad Style) """

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        # return ''.join(ch for ch in text if ch not in exclude)
        return ''.join(' ' if ch in exclude else ch for ch in text)

    def lower(text):
        text_lower = list()
        for i in text:
            text_lower.append(i.lower())
        return text_lower

    #         return text.lower()
    s_normalized = white_space_fix(remove_articles(remove_punc(lower(s))))
    if " ’s" in s_normalized:
        s_normalized = s_normalized.replace(" ’s", "s")
    return s_normalized


def eval_text_f1_score(gt_dict_word, pred_dict_word, arg_list, gt_data):
    gt_num, pred_num, correct_num = 0, 0, 0
    gt_num_identify, pred_num_identify, correct_identify_num = 0, 0, 0

    for i in range(len(pred_dict_word)):
        all_pred_list = list()
        all_gt_list = list()
        for arg_role in arg_list[gt_data[i][2]]:
            pred_text_list = pred_dict_word[i][arg_role] if arg_role in pred_dict_word[i].keys() else list()
            gt_text = gt_dict_word[i][arg_role] if arg_role in gt_dict_word[i].keys() else list()
            for pred_text in pred_text_list:
                pred_text = _normalize_answer(pred_text)
                for gt in gt_text:
                    temp = list()
                    if type(gt) == str:
                        gt = [gt]
                    for t in gt:
                        if t != []:
                            temp.append(_normalize_answer(t))
                    if pred_text in temp:
                        correct_num += 1

                if pred_text != 'not specified' and pred_text != '':
                    pred_num += 1
                    all_pred_list.append(pred_text)
            for gt in gt_text:
                temp = list()
                if type(gt) == str:
                    gt = [gt]
                for t in gt:
                    if t != []:
                        temp.append(_normalize_answer(t))
                if gt != []:
                    all_gt_list.append(temp)

        gt_num += len(all_gt_list)
        # print('pred:',all_pred_list)
        # print('gt1:',all_gt_list, len(all_gt_list))
        all_gt_list = [i for item in all_gt_list for i in item]

        all_pred_list = list(all_pred_list)
        all_gt_list = list(all_gt_list)
        # print("gt2:", all_gt_list, len(all_gt_list))
        for gt_span in all_gt_list:
            if gt_span in all_pred_list:
                correct_identify_num += 1

    res_classification = eval_rpf(gt_num, pred_num, correct_num)
    res_identification = eval_rpf(gt_num, pred_num, correct_identify_num)
    return res_classification, res_identification

def acc_evaluation(predictions, gt_data, data_type):
    arg_dict = dict()
    for i in gt_data:
        arg_word = list()
        if data_type == 'cross':
            arg_list = json.loads(i[3])
        elif data_type == 'normal':
            arg_list = i[3]
        for arg in arg_list:
            arg_word.append(arg['type'])
        arg_word = list(set(arg_word))
        if i[2] not in arg_dict.keys():
            arg_dict.update({i[2]: arg_word})
        else:
            for arg in arg_word:
                if arg not in arg_dict[i[2]]:
                    value_list = arg_dict[i[2]]
                    value_list.extend([arg])
                    arg_dict.update({i[2]: value_list})

    gt_word_docee = list()

    for i in gt_data:
        gt_dict = defaultdict(list)
        if data_type == 'normal':
            arg_list = i[3]
            for j in arg_list:
                arg = j['type']
                text = [j['text']]
                if arg in gt_dict.keys():
                    gt_dict[arg].append(text)
                else:
                    gt_dict[arg] = text
        if data_type == 'cross':
            arg_list = json.loads(i[3])
            for j in arg_list:
                arg = j['type']
                text = list()
                for t in j["mention"]:
                    text.append(t["text"])
                if arg in gt_dict.keys():
                    gt_dict[arg].extend(text)
                else:
                    gt_dict[arg] = text
        gt_word_docee.append(gt_dict)

    pred_dict_word = predictions
    acc_classification, acc_identification = eval_text_f1_score(gt_word_docee, pred_dict_word, arg_dict, gt_data)
    return acc_classification, acc_identification
