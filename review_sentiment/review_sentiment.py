import pickle
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import numpy as np
import random
import os

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

from transformers import logging
logging.set_verbosity_error()

path = "./reviewdataset/review_posneg_theme_byjob/"
file_list = os.listdir(path)
jobs = ['기획경영','개발','마케팅시장조사']
for filename in file_list:
    jobname = filename.split('_')[0]
    if len(jobs)==0:
        jobs.append(jobname)
    if jobname != jobs[-1]:
        jobs.append(jobname)

keywords = ['balance', 'salary', 'benefit', 'environment']
model_name = "kykim/bert-kor-base"

def review_sentiment(keyword, job):
    # check
    print(keyword, job)

    data_by_keyword = {}
    combined = {}

    with open(f'./reviewdataset/review_posneg_theme_byjob/{job}_pos_{keyword}.pkl', 'rb') as t:
        dev_pos_data = pickle.load(t)
        data_by_keyword[f'{job}_pos_{keyword}_data'] = dev_pos_data

    with open(f'./reviewdataset/review_posneg_theme_byjob/{job}_neg_{keyword}.pkl', 'rb') as f:
        dev_neg_data = pickle.load(f)
        data_by_keyword[f'{job}_neg_{keyword}_data'] = dev_neg_data

    combined[f'_{job}_{keyword}_data'] = data_by_keyword[f'{job}_pos_{keyword}_data'] + data_by_keyword[f'{job}_neg_{keyword}_data']
    random.shuffle(combined[f'_{job}_{keyword}_data'])

    second_values = []
    positive_value= []
    negative_value = []
    
    for data_point in combined[f'_{job}_{keyword}_data']:
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        tokenizer = BertTokenizer.from_pretrained(model_name)

        model.eval()

        tokens = tokenizer(data_point, padding=True, truncation=True, return_tensors="pt")
        
        with torch.no_grad():
            prediction = model(**tokens)

        prediction = F.softmax(prediction.logits, dim=1)
        prediction_new = [item[1].item() for item in prediction]

        for pred in prediction_new:
            second_values.extend([item[1].item() for item in prediction])
            if pred >= 0.5:
                positive_value.extend([item[1].item() for item in prediction])
            else:
                negative_value.extend([item[1].item() for item in prediction])

    po_avg = np.mean(positive_value)
    ne_avg = np.mean(negative_value)

    print(po_avg)
    print(ne_avg)

    path = './result'
    makedirs(path)

    with open(f'{path}/{job}_{keyword}.txt', 'w') as file:
        for value in second_values:
            file.write(f'{value}\n')
    with open(f'{path}/{job}_{keyword}_po_avg.txt', 'w') as file:
            file.write(f'{po_avg}')
    with open(f'{path}/{job}_{keyword}_ne_avg.txt', 'w') as file:
            file.write(f'{ne_avg}')

# run
for job in jobs:
    for keyword in keywords:
        review_sentiment(keyword, job)