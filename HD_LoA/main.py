import argparse
import subprocess
from RAMS.main_RAMS import *
from DocEE.main_DocEE import *

api_key = "" # put your openai api key here

def run_experiment(dataset_name, model_type, data_type):
    if dataset_name == "RAMS":
        run_RAMS(model_type, api_key)
    elif dataset_name == "DocEE":
        run_DocEE(model_type, data_type, api_key)


def main():
    parser = argparse.ArgumentParser(description="Run experiments based on the dataset")
    parser.add_argument("--dataset_name", type=str, choices=['RAMS', 'DocEE'], help="The name of the dataset", default='DocEE')
    parser.add_argument("--data_type", type=str, choices=['normal', 'cross'], help="The type of the docEE dataset", default='normal')
    parser.add_argument("--model_type", type=str, choices=['text-davinci-003', 'gpt-3.5-turbo-instruct', 'gpt-4'], help="The name of the model", default='gpt-3.5-turbo-instruct')

    args = parser.parse_args()
    run_experiment(args.dataset_name, args.model_type, args.data_type)


if __name__ == "__main__":
    main()
