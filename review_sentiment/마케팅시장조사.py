import pickle
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

keyword = ['balance', 'salary', 'benefit', 'environment']

model_name = "kykim/bert-kor-base"
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = BertTokenizer.from_pretrained(model_name)

for kw in keyword:
    print(kw)
    data_by_keyword = {}
    combined = {}

    with open(f'./reviewdataset/review_posneg_theme_byjob/마케팅시장조사_pos_{kw}.pkl', 'rb') as t:
        마케팅시장조사_pos_data = pickle.load(t)
        data_by_keyword[f'마케팅시장조사_pos_{kw}_data'] = 마케팅시장조사_pos_data

    with open(f'./reviewdataset/review_posneg_theme_byjob/마케팅시장조사_neg_{kw}.pkl', 'rb') as f:
        마케팅시장조사_neg_data = pickle.load(f)
        data_by_keyword[f'마케팅시장조사_neg_{kw}_data'] = 마케팅시장조사_neg_data

    combined[f'_마케팅시장조사_{kw}_data'] = data_by_keyword[f'마케팅시장조사_pos_{kw}_data'] + data_by_keyword[f'마케팅시장조사_neg_{kw}_data']
    print(len(combined[f'_마케팅시장조사_{kw}_data']))

    second_values = []
    for data_point in combined[f'_마케팅시장조사_{kw}_data']:
        tokens = tokenizer(data_point, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            prediction = model(**tokens)

        prediction = F.softmax(prediction.logits, dim=1)
        second_values.extend([item[1].item() for item in prediction])
        average_second_value = sum(second_values) / len(second_values)
    
    print(len(second_values))
    print(average_second_value)

    with open(f'./review_sentiment/sentiment_values/마케팅시장조사_{kw}_second_values.txt', 'w') as file:
        for value in second_values:
            file.write(f'{value}\n')

    with open(f'./review_sentiment/sentiment_values/마케팅시장조사_{kw}_average_second_values.txt', 'w') as file:
        file.write(f'Average Second Value for 마케팅시장조사 {kw}: {average_second_value}')