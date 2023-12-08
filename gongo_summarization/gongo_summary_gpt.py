import os
import openai
from konlpy.tag import Okt
okt = Okt()

openai.api_key = "enter the api key"

import pickle
file_path = '..\\gongodataset\\gongo_workqual_byjob\\qual_all.txt'

with open(file_path, 'rb') as t:
    text = pickle.load(t)

text = ' '.join(text)

max_length=3000
if len(text) > max_length:
    text = text[:max_length]

directive = """
    - This is a chatbot that summarizes the recruitment document of companies.
    - The result should be in Korean.
    - Summarize in terms of major task, qualificaion or preference.
    - Here is the business reviews: {}""".format(text)

query = 'Summarize the recruitment document in one sentence'

response = openai.ChatCompletion.create(
    model = "gpt-3.5-turbo",
    messages = [
        {"role": "system", "content": directive},
        {"role": "user", "content": query},
    ]
)

output_text = response["choices"][0]["message"]["content"]
print(output_text)