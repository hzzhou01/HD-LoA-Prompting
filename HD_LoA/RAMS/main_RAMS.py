import os
import openai
import tiktoken
import time
import json
import random
from tqdm import tqdm
from .data.prompt_generator import *
from .data.evaluate_RAMS import *
random.seed(0)

def run_RAMS(MODEL_TYPE, API_KEY):
    # Configuration
    DATA_FILE = './RAMS/data/RAMS_1.0/data/test.jsonlines'
    ARGUMENTS_FILE = './RAMS/data/RAMS_1.0/data/arg_roles_rams_concat.csv'
    DEMON_FILE = './RAMS/data/demonstration.txt'
    LOG_FILE = f"./RAMS/log/RAMS_{MODEL_TYPE}.txt"
    OUTPUT_FILE = f"./RAMS/log/RAMS_{MODEL_TYPE}.jsonlines"

    openai.api_key = API_KEY
    encoder = tiktoken.encoding_for_model(MODEL_TYPE)

    data_list = data_loader(DATA_FILE)
    # due to the cost of GPT4, only 200 samples are evaluated
    if MODEL_TYPE == 'gpt-4':
        data_list = random.sample(data_list, 200)

    pred_all = []
    for index, data_i in enumerate(tqdm(data_list)):
        prompt = prompt_generator(DEMON_FILE, ARGUMENTS_FILE, data_i)
        gpt_output = gen_gpt_output(prompt, MODEL_TYPE, encoder)
        output_log = f"sample{index}\n{gpt_output}\n\n\n\n"
        write_output_file(LOG_FILE, output_log)
        answer_dict = ans_extraction(gpt_output)

        if not answer_dict:  # gpt-3.5 model may not always follow instructions due to its limited reasoning ability
            prompt += "\nElaborate the meaning of event type and its argument roles:"
            gpt_output = gen_gpt_output(prompt, MODEL_TYPE, encoder)
            output_log = f"sample{index}\n{gpt_output}\n\n\n\n"
            write_output_file(LOG_FILE, output_log)
            answer_dict = ans_extraction(gpt_output)

        with open(OUTPUT_FILE, 'a') as f:
            f.write(json.dumps(answer_dict) + '\n')

        pred_all.append(answer_dict)
        if index % 100 == 0:
            acc_classification, acc_identification = acc_evaluation(pred_all, data_list)
            print(f"acc:\n{acc_classification}\n{acc_identification}")
    acc_classification, acc_identification = acc_evaluation(pred_all, data_list)
    print(f"acc:\n{acc_classification}\n{acc_identification}")

def gen_gpt_output(prompt, model_type, encoder):
    api_time_interval = 1
    time.sleep(api_time_interval)
    while True:
        try:
            gpt_output = evaluate_prompt(prompt, model_type, encoder)
            break
        except Exception as e:
            print(f"An error occurred: {e}. Retrying in 1 second...")
            time.sleep(1)
    return gpt_output

def ans_extraction(answer):
    answer = '\n' + answer
    answer_dict = {}
    extracted_arguments = extract_strings(answer, start_string="]: \"", end_string="\"")
    argument_roles = extract_strings(answer, start_string="\n[", end_string="]:")
    for i in range(len(extracted_arguments)):
        role = argument_roles[i]
        argument = extracted_arguments[i]
        answer_dict[role] = argument
    return answer_dict

def evaluate_prompt(prompt, model, encoder):
    prompt_len = len(encoder.encode(prompt)) + 20
    if model == "text-davinci-003":
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
            if response["usage"]["completion_tokens"] == 4096-prompt_len:
                print('sample unfinished')
            return response["choices"][0]["text"]
    if model == "gpt-3.5-turbo-instruct":
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
            if response["usage"]["completion_tokens"] == 4096-prompt_len:
                print('sample unfinished')
            return response["choices"][0]["text"]
    elif model == "gpt-3.5-turbo":
        message = [{"role":"user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=message,
            temperature=0,
            max_tokens=max(4096 - prompt_len,0),
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        if response["usage"]["completion_tokens"] == 4096 - prompt_len:
            print('sample unfinished')
    elif model == "gpt-4":
        message = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=message,
            temperature=0,
            max_tokens=2500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
    if response["usage"]["completion_tokens"] == 4096 - prompt_len:
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