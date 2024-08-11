import jsonlines
import json
import csv
import re
import spacy
import random
import openai
import tiktoken
random.seed(0)
max_doc_len = 1250
question_base = 'Question: Extract the event arguments of'
question_inst = 'When pinpointing each event argument, it\'s crucial to quote the entity exactly as it appears in the text. Note that if an event argument is not explicitly mentioned or cannot be directly associated with its argument role in question, please respond with \"not specified\".'

nlp = spacy.load("en_core_web_sm")

model_type = "gpt-3.5-turbo-instruct"
encoder = tiktoken.encoding_for_model(model_type)

def remove_symbols(lst):
    return [item for item in lst if str(item).isalnum()]

def data_loader(data_file_name, data_type, rand_sample_num):
    with jsonlines.open(data_file_name) as f:
        data_list = list(f)[0]
        filtered_data_list = filter_data_list(data_list)
        # random select n samples for evaluation
        rand_index = [random.randint(0, len(filtered_data_list) - 1) for _ in range(rand_sample_num)]
        rand_select_datalist = [filtered_data_list[i] for i in rand_index]

        # build argument roles dict for each event
        arg_dict = dict()
        if data_type == 'normal':
            for i in data_list:
                arg_word = list()
                for arg in i[3]:
                    arg_word.append(arg['type'])
                arg_word = list(set(arg_word))
                if i[2] not in arg_dict.keys():
                    arg_dict.update({i[2]:arg_word})
                else:
                    for arg in arg_word:
                        if arg not in arg_dict[i[2]]:
                            value_list = arg_dict[i[2]]
                            value_list.extend([arg])
                            arg_dict.update({i[2]:value_list})
        elif data_type == 'cross':
            for i in data_list:
                line = json.loads(i[3])
                arg_word = list()
                for arg in line:
                    arg_word.append(arg["type"])
                arg_word = list(set(arg_word))
                if i[2] not in arg_dict.keys():
                    arg_dict.update({i[2]: arg_word})
                else:
                    for arg in arg_word:
                        if arg not in arg_dict[i[2]]:
                            value_list = arg_dict[i[2]]
                            value_list.extend([arg])
                            arg_dict.update({i[2]: value_list})
    return rand_select_datalist, arg_dict

def filter_data_list(data_list):
    # Filter out samples that are too long and may exceed the maximum context length
    filtered_data_list = []
    for data_i in data_list:
        doc_i = data_i[1]
        doc_i = doc_i.replace('\n', ' ')
        doc_len = len(encoder.encode(doc_i))
        if doc_len < max_doc_len:
            filtered_data_list.append(data_i)
    return filtered_data_list


def prompt_generator(demon_file, data_i, arg_dict, arg_num, arg_index = 0, option='arg_split'):
    with open(demon_file, "r") as f1:
        demon = f1.read()

    event = data_i[2]
    doc_i = data_i[1]
    title = data_i[0]
    doc_i = re.sub(r'\n+', '\n', doc_i)
    doc_i = doc_i.strip()

    if option == 'arg_split':
        # Split argument roles into subsets to avoid exceeding the maximum content length of the LLM (for docEE data sets, an event could consist of dozens of arg roles)
        arguments = arg_dict[event]
        arg_subset = arguments[arg_index:arg_index + arg_num]
        formatted_args = "\', \'".join(arg_subset)
        event = event.replace("n/a", "na")
        # question_1 = f" '{formatted_args}' in the '{event}' event in the provided news document below."
        query = f"{question_base} '{formatted_args}' in the '{event}' event in the provided news document below. {question_inst}"
        prompt = f"{demon}\n{query}\n\nDocument:\n{doc_i}\n\nAnswer (adapting the format of the answer in the example):"
        arg_index += arg_num
        if arg_index >= len(arguments):
            arg_index = None
    elif option == 'arg_full':
        # Not split argument roles into subsets
        arguments = arg_dict[event]
        arg_subset = arguments
        formatted_args = "\', \'".join(arguments)
        event = event.replace("n/a", "na")
        # question_1 = f" '{formatted_args}' in the '{event}' event in the provided news document below."
        query = f"{question_base} '{formatted_args}' in the '{event}' event in the provided news document below. {question_inst}"
        prompt = f"{demon}\n{query}\n\nDocument:\n{doc_i}\n\nAnswer (adapting the format of the answer in the example):"
        arg_index = None
        # with open(output_file, "a") as outfile:
        #     # Write the string to the file
        #     outfile.write(prompt)
        #     outfile.write('\n\n\n\n')
    return prompt, arg_index, arg_subset


