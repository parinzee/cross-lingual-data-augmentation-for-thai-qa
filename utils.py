import random
import torch
import numpy as np
import hashlib
from collections import defaultdict

def seed_everything(seed=40):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def convert_row_to_simple_transformers_format(row, question_col="question"):
    # Initialize an empty list to store converted answers
    converted_answers = []

    # Iterate over each answer and its corresponding start index
    for i in range(len(row['answers']['text'])):
        # Create a dictionary for the current answer
        answer_dict = {
            'text': row['answers']['text'][i],
            'answer_start': row['answers']['answer_start'][i],
        }

        # Add the current answer to the list of converted answers
        converted_answers.append(answer_dict)

    # Simpletransformers requires that ids be unique
    # If we augment the questions, the current id scheme would collide
    # Thus we rehash the id with the current question instead
    id = hashlib.sha256((row["context"] + row[question_col]).encode("utf-8")).hexdigest()

    # Create a dictionary for the question and answers
    qas_dict = {
        'id': id,
        "is_impossible": False,
        'question': row[question_col],
        'answers': converted_answers
    }

    # Wrap the 'context', 'question', and 'answers' into a 'qas' list
    # Return the converted example
    return {'context': row['context'], 'qas': [qas_dict]}

def merge_qas(data):
    merged_data = defaultdict(list)

    # Loop over each dictionary in the dataset
    for item in data:
        # Add each 'qas' to the list of 'qas' for the same 'context'
        merged_data[item['context']].extend(item['qas'])

    # Convert the merged_data back into the original format: a list of dictionaries
    merged_data = [{'context': context, 'qas': qas} for context, qas in merged_data.items()]

    return merged_data