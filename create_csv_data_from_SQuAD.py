import pandas as pd
import json

from constants import BASE_PATH

paragraphs = [] 
questions = []

with open(BASE_PATH/"data/train-v1.1.json", 'r') as file:
    train_squad_data = json.load(file)

with open(BASE_PATH/"data/dev-v1.1.json", 'r') as file:
    valid_squad_data = json.load(file)

train_dict = {'source': [], 'target': []}
validation_dict = {'source': [], 'target': []}

for article in train_squad_data['data']:
    for paragraph in article['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            question = qa['question']
            train_dict['source'].append(context)
            train_dict['target'].append(question)


for article in valid_squad_data['data']:
    for paragraph in article['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            question = qa['question']
            validation_dict['source'].append(context)
            validation_dict['target'].append(question)


train_df = pd.DataFrame(train_dict)
validation_df = pd.DataFrame(validation_dict)

train_df.to_csv(BASE_PATH/"data/train_squad.csv", index=False)
validation_df.to_csv(BASE_PATH/"data/valid_squad.csv", index=False)

