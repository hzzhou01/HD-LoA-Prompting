import jsonlines
import csv
import re
import spacy

# Configuration
arguments_file_2 = './RAMS/data/RAMS_1.0/data/prompts_rams_full.csv'
question_base = 'Question: Extract the event arguments of'
question_inst  = 'When pinpointing each event argument, it\'s crucial to quote the entity exactly as it appears in the text. If an event argument is not explicitly mentioned or cannot be directly associated with the event indicated by the trigger word, please respond with "not specified".'
nlp = spacy.load("en_core_web_sm")

def remove_symbols(lst):
    return [item for item in lst if str(item).isalnum()]

def data_loader(data_file_name):
    """Load data from a jsonlines file."""
    with jsonlines.open(data_file_name) as f:
        data_list = list(f)
    return data_list

def prompt_generator(demon_file, arguments_file, data_i):
    """Generate prompt from given data."""
    with open(demon_file, "r") as f:
        demon = f.read()

    event = data_i['evt_triggers'][0][2][0][0]
    tri_index = data_i['evt_triggers'][0][0]
    sentences = data_i['sentences']
    trigger = insert_trigger_tags(sentences, tri_index) # insert special tokens <t> </t> for trigger word
    document_text = " ".join([" ".join(sentence) for sentence in data_i['sentences']])
    trigger_sent = extract_trigger_sentence(document_text)
    arguments = get_event_arguments(arguments_file, arguments_file_2, event)
    arguments_description = format_arguments(arguments)

    question = f"{question_base} {arguments_description} in the \"{event.replace('n/a','na')}\" event in the provided document, with the trigger word being \"{trigger}\", highlighted between \"<t>\" and \"</t>\" in the news document. "
    question = question + question_inst
    prior = 'Prioritize the identification of event arguments within the specified trigger sentence. If an event argument is not explicitly mentioned, please answer "not specified".'

    # print(demon + '\n\n' + document_text +'\n\n' + question)
    prompt_text = (
        f"{demon}\n{question}\n\n"
        f"Document: {document_text}\n\n"
        f"Trigger sentence: \"{trigger_sent}\"\n"
        f"{prior}\n\n"
        # f"\n"
        "Answer:"
    )
    return prompt_text

def insert_trigger_tags(sentences, tri_index):
    """Insert special trigger tags around the trigger word in sentences."""
    count = -1
    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            count += 1
            if count == tri_index:
                trigger = sentences[i][j]
                # Insert the <t> and </t> tags before and after the word
                sentences[i][j] = "<t> " + sentences[i][j] + " </t>"
    return trigger

def extract_trigger_sentence(document_text):
    trigger_sent = ''
    sent_list = [sent.text for sent in nlp(document_text).sents]
    pattern = r'<t>.*</t>'
    for sent in sent_list:
        if re.search(pattern, sent):
            trigger_sent = sent
            break
    return trigger_sent

def get_event_arguments(arguments_file, arguments_file_2, event):
    """Retrieve event arguments for a given event type"""
    with open(arguments_file, 'r') as f2:
        reader = csv.reader(f2)
        # Search for keywords in each row
        for row in reader:
            event_type = re.split(r'[ :]', row[0])[0]
            if event == event_type:
                arguments = re.split(r'[ :]', row[0])[1:]
                arguments = remove_symbols(arguments)
                arguments = list(set(arguments))

    with open(arguments_file_2, 'r') as f3:
        reader2 = csv.reader(f3)
        for row2 in reader2:
            event_type2 = re.split(r'[:]',row2[0])[0]
            if event == event_type2:
                arg_sentence = re.split(r'[:]', ' '.join(row2))[1:][0]
                arguments.sort(key=lambda x: arg_sentence.split().index(x))
    return arguments

def format_arguments(arguments):
    """Format argument list into a string with proper grammar for the prompt."""
    if len(arguments) > 1:
        arguments[-1] = "and " + arguments[-1]
        arguments_seq = ", ".join(arguments)
    else:
        arguments_seq = arguments[0]
    return arguments_seq
