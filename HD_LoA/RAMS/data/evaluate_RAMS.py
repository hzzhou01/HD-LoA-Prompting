import os
import json
import copy
from itertools import chain
import csv
import string
import re
import logging


logger = logging.getLogger(__name__)

def _read_roles(role_path):
    template_dict = {}
    role_dict = {}

    with open(role_path, "r", encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for line in csv_reader:
            event_type_arg, template = line
            template_dict[event_type_arg] = template
                
            event_type, arg = event_type_arg.split('_')
            if event_type not in role_dict:
                role_dict[event_type] = []
            role_dict[event_type].append(arg)

    return template_dict, role_dict

template_dict, argument_dict = _read_roles("./RAMS/data/RAMS_1.0/data/description_rams.csv")
output_dir = './RAMS/log'
class Event:
    def __init__(self, doc_id, sent_id, sent, event_type, event_trigger, event_args, full_text, first_word_locs=None):
        self.doc_id = doc_id
        self.sent_id = sent_id
        self.sent = sent
        self.type = event_type
        self.trigger = event_trigger
        self.args = event_args
        
        self.full_text = full_text
        self.first_word_locs = first_word_locs


    def __str__(self):
        return self.__repr__()
    

    def __repr__(self):
        s = ""
        s += "doc id: {}\n".format(self.doc_id)
        s += "sent id: {}\n".format(self.sent_id)
        s += "text: {}\n".format(" ".join(self.sent))
        s += "event_type: {}\n".format(self.type)
        s += "trigger: {}\n".format(self.trigger['text'])
        for arg in self.args:
            s += "arg {}: {} ({}, {})\n".format(arg['role'], arg['text'], arg['start'], arg['end'])
        s += "----------------------------------------------\n"
        return s

def _create_example_rams(lines):
        invalid_arg_num = 0
        W = 250
        assert(W%2==0)
        all_args_num = 0

        examples = []
        for line in lines:
            if len(line["evt_triggers"]) == 0:
                continue
            doc_key = line["doc_key"]
            events = line["evt_triggers"]

            full_text = copy.deepcopy(list(chain(*line['sentences'])))
            cut_text = list(chain(*line['sentences']))
            sent_length = sum([len(sent) for sent in line['sentences']])

            text_tmp = []
            first_word_locs = []
            for sent in line["sentences"]:
                first_word_locs.append(len(text_tmp))
                text_tmp += sent

            for event_idx, event in enumerate(events):                
                event_trigger = dict()
                event_trigger['start'] = event[0]
                event_trigger['end'] = event[1]+1
                event_trigger['text'] = " ".join(full_text[event_trigger['start']:event_trigger['end']])
                event_type = event[2][0][0]

                offset, min_s, max_e = 0, 0, W+1
                event_trigger['offset'] = offset
                if sent_length > W+1:
                    if event_trigger['end'] <= W//2:     # trigger word is located at the front of the sents
                        cut_text = full_text[:(W+1)]
                    else:   # trigger word is located at the latter of the sents
                        offset = sent_length - (W+1)
                        min_s += offset
                        max_e += offset
                        event_trigger['start'] -= offset
                        event_trigger['end'] -= offset 
                        event_trigger['offset'] = offset
                        cut_text = full_text[-(W+1):]

                event_args = list()
                for arg_info in line["gold_evt_links"]:
                    if arg_info[0][0] == event[0] and arg_info[0][1] == event[1]:  # match trigger span    
                        all_args_num += 1

                        evt_arg = dict()
                        evt_arg['start'] = arg_info[1][0]
                        evt_arg['end'] = arg_info[1][1]+1
                        evt_arg['text'] = " ".join(full_text[evt_arg['start']:evt_arg['end']])
                        evt_arg['role'] = arg_info[2].split('arg', maxsplit=1)[-1][2:]
                        if evt_arg['start']<min_s or evt_arg['end']>max_e:
                            invalid_arg_num += 1
                        else:
                            evt_arg['start'] -= offset
                            evt_arg['end'] -= offset 
                            event_args.append(evt_arg)

                if event_idx > 0:
                    examples.append(Event(doc_key+str(event_idx), None, cut_text, event_type, event_trigger, event_args, full_text, first_word_locs))
                else:
                    examples.append(Event(doc_key, None, cut_text, event_type, event_trigger, event_args, full_text, first_word_locs))
            
        # logger.info("{} examples collected. {} arguments dropped.".format(len(examples), invalid_arg_num))

        return examples

def eval_rpf(gt_num, pred_num, correct_num):
    recall = correct_num/gt_num if gt_num!=0 else .0
    precision = correct_num/pred_num if pred_num!=0 else .0
    f1 = 2*recall*precision/(recall+precision) if (recall+precision)>1e-4 else .0
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

def eval_text_f1_score(gt_dict_word, pred_dict_word, arg_list, invalid_gt_num=0):
    gt_num, pred_num, correct_num = 0, 0, 0
    gt_num_identify, pred_num_identify, correct_identify_num = 0, 0, 0

    for i in range(len(pred_dict_word)):
    # for i in range(17):
        all_pred_list = list()
        all_gt_list = list()
        for arg_role in arg_list[i]:
            pred_text = pred_dict_word[i][arg_role] if arg_role in pred_dict_word[i].keys() else list()
            gt_text = gt_dict_word[i][arg_role] if arg_role in gt_dict_word[i].keys() else list()
            gt_text = _normalize_answer(gt_text)
            pred_text = _normalize_answer(pred_text)

            if pred_text != 'not specified' and pred_text != '':
                pred_num += 1
                all_pred_list.append(pred_text)

            if gt_text != '':
                gt_num += 1
                all_gt_list.append(gt_text)
            
            if gt_text != '' and gt_text == pred_text:
                correct_num += 1

#         print('pred:',all_pred_list)
#         print('gt:',all_gt_list)
        all_pred_list = list(set(all_pred_list))
        all_gt_list = list(set(all_gt_list))
        pred_num_identify += len(all_pred_list)
        gt_num_identify += len(all_gt_list)
        for gt_span in all_gt_list:
            if gt_span in all_pred_list:
                correct_identify_num += 1
        
    res_classification = eval_rpf(gt_num+invalid_gt_num, pred_num, correct_num)
    res_identification = eval_rpf(gt_num_identify+invalid_gt_num, pred_num_identify, correct_identify_num)
    return res_classification, res_identification

def acc_evaluation(predictions, gt_data):
    arg_list = list()
    
    gt_rams = _create_example_rams(gt_data)
    gt_dict_word = list()

    for i in gt_rams:
        gt_arg = dict()
        for j in i.args:
            gt_arg.update({j['role']:j['text']})
        gt_dict_word.append(gt_arg)

    for i in gt_rams:
        event_type = i.type
        arg_dict = argument_dict[event_type.replace(':', '.')]
        arg_list.append(arg_dict)
    # print(arg_list)

    pred_dict_word = predictions

    acc_classification, acc_identification = eval_text_f1_score(gt_dict_word, pred_dict_word, arg_list)
    return acc_classification, acc_identification

