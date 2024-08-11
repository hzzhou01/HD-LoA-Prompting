import os
import openai
import tiktoken
import time
import json
import difflib
import jsonlines
import random
random.seed(0)
from .data.evaluate_docEE import *
from .data.prompt_generator import *

def compare_strings(s1, s2):
    d = difflib.Differ()
    diff = d.compare(s1, s2)
    print('\n'.join(diff))


def run_DocEE(model_type, data_type, api_key):
    # Configuration
    if data_type == 'normal':
        data_file_name = './DocEE/data/normal_setting/test.json'
    elif data_type == 'cross':
        data_file_name = './DocEE/data/cross_domain_setting/test_targe_domain.json'
    else:
        raise Exception("Failed to indicate data type for DocEE")
    demon_file = './DocEE/data/demonstration_docEE.txt'
    output_log_file = f"./DocEE/log/docEE_{data_type}_{model_type}.txt"
    outputfile = f"./DocEE/log/docEE_{data_type}_{model_type}.jsonlines"

    openai.api_key = api_key
    encoder = tiktoken.encoding_for_model(model_type)

    # Randomly select 100 samples for gpt-4, 400 samples for other LLMs, for cross and normal setting respectively
    if model_type == 'gpt-4':
        rand_sample_num = 100
    else:
        rand_sample_num = 400
    data_list, arg_dict = data_loader(data_file_name, data_type, rand_sample_num)
    index = 0
    pred_all = []
    for data_i in data_list:
        index += 1
        arg_index = 0
        answer_dict = {}
        while arg_index!=None:
            if model_type == "gpt-4":
                prompt, arg_index, arg_subset = prompt_generator(demon_file, data_i, arg_dict, arg_num=8, arg_index=arg_index)
            else:
                prompt, arg_index, arg_subset = prompt_generator(demon_file, data_i, arg_dict, arg_num=5, arg_index=arg_index)
            print(f"{index}: {arg_index}")

            gpt_output = gen_gpt_output(prompt, model_type, encoder)
            output_log = "sample" + str(index) + "\n" + gpt_output + "\n\n\n\n"
            write_output_file(output_log_file, output_log)
            if model_type == 'gpt-4':
                answer_dict_i = ans_extraction_gpt4(gpt_output, arg_subset)
            else:
                answer_dict_i = ans_extraction(gpt_output, arg_subset)
            answer_dict.update(answer_dict_i)
            if len(answer_dict_i) == 0: #  gpt-3.5 model may not always follow instructions due to its limited reasoning ability
                prompt += "\nElaborate the meaning of event type and its argument roles:"
                gpt_output = gen_gpt_output(prompt, model_type)
                output_log = "sample" + str(index) + "\n" + gpt_output + "\n\n\n\n"
                write_output_file(output_log_file, output_log)
                if model_type == 'gpt-4':
                    answer_dict_i = ans_extraction_gpt4(gpt_output, arg_subset)
                else:
                    answer_dict_i = ans_extraction(gpt_output, arg_subset)
                answer_dict.update(answer_dict_i)
            if model_type != "gpt-4":
                if len(answer_dict_i) < len(arg_subset):
                    if arg_index:
                        arg_index = arg_index - (len(arg_subset)-len(answer_dict_i))
                    else:
                        arg_index = len(arg_dict[data_i[2]]) - (len(arg_subset) - len(answer_dict_i))
        with open(outputfile, 'a') as f:
            f.write(json.dumps(answer_dict))
            f.write('\n')
        pred_all.append(answer_dict)
        # acc_evaluation(pred_all, data_list, data_type)
        # acc_evaluation(pred_all_2, data_list, data_type)
        if index % 100 ==0:
            acc_classification, acc_identification = acc_evaluation(pred_all,data_list, data_type)
            print(f"acc:\n{acc_classification}\n{acc_identification}")
    acc_classification, acc_identification = acc_evaluation(pred_all, data_list, data_type)
    print(f"acc:\n{acc_classification}\n{acc_identification}")

def ans_extraction(answer, arg_subset):
    answer = '\n' + answer + '\n'  # for the extraction of arguments and roles from output text
    answer_dict_i = {}
    extracted_arguments = extract_strings(answer, start_string="]: \"", end_string="\"\n")
    argument_roles = arg_subset[:len(extracted_arguments)]
    extract_num = min(len(arg_subset), len(extracted_arguments))
    for i in range(extract_num):
        role = argument_roles[i]
        argument = extracted_arguments[i].split('\", \"')[:3]
        answer_dict_i[role] = argument
    return answer_dict_i

def ans_extraction_gpt4(answer, arg_subset):
    answer = '\n' + answer + '\n'
    answer_dict_i = {}
    extracted_arguments = extract_strings(answer, start_string="]: \"", end_string="\"\n")
    argument_roles = extract_strings(answer, start_string="\n[", end_string="]:")[:len(extracted_arguments)]
    extract_num = min(len(arg_subset), len(extracted_arguments))
    for i in range(extract_num):
        role = argument_roles[i]
        argument = extracted_arguments[i].split('\", \"')[:3]
        answer_dict_i[role] = argument
    return answer_dict_i

def gen_gpt_output(prompt, model_type, encoder):
    api_time_interval = 1
    time.sleep(api_time_interval)
    while True:
        try:
            gpt_output = evaluate_prompt(prompt, model_type, encoder)
            break
        except Exception as e:
            print(f"An error occurred: {e}. Retrying in one second...")
            time.sleep(1)
    return gpt_output

def split_content(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()

        # Split the content by the delimiter
        split_content = content.split("\n\n\n\n")
        return split_content
    except IOError:
        print("An error occurred while reading the file.")
        return None

def evaluate_prompt(prompt, model, encoder):
    if model == "text-davinci-003":
        prompt_len = len(encoder.encode(prompt)) + 20
        if prompt_len>4096:
            return 'None'
        else:
            max_tokens = max(4096 - prompt_len,0)
            response = openai.Completion.create(
                model=model,
                prompt=prompt,
                temperature=0,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            if response["usage"]["completion_tokens"] == max_tokens:
                print('sample unfinished')
            return response["choices"][0]["text"]
    elif model == "gpt-3.5-turbo-instruct":
        prompt_len = len(encoder.encode(prompt)) + 20
        if prompt_len>4096:
            return 'None'
        else:
            max_tokens = max(4096 - prompt_len,0)
            response = openai.Completion.create(
                model=model,
                prompt=prompt,
                temperature=0,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            if response["usage"]["completion_tokens"] == max_tokens:
                print('sample unfinished')
            return response["choices"][0]["text"]
    elif model == "gpt-4":
        engine = model
        prompt_len = len(encoder.encode(prompt)) + 20
        message = [{"role": "user", "content": prompt}]
        max_tokens = 4096
        response = openai.ChatCompletion.create(
            model=engine,
            messages=message,
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        if response["usage"]["completion_tokens"] == max_tokens:
            print('sample unfinished')
    return response["choices"][0]["message"]["content"]

def write_output_file(output_file, content):
    with open(output_file, "a") as outfile:
        # Write the string to the file
        outfile.write(content)

def extract_strings(input_string, start_string, end_string):
    extracted_strings = []

    start_index = 0
    while True:
        start_index = input_string.find(start_string, start_index)
        if start_index == -1:
            break

        start_index += len(start_string)
        end_index = input_string.find(end_string, start_index)

        if end_index == -1:
            break

        extracted_string = input_string[start_index:end_index]
        extracted_strings.append(extracted_string)

        start_index = end_index

    return extracted_strings

if __name__ == "__main__":
    main()