import os
import copy
import json
import random
import csv
import re
import numpy as np
random.seed(0)

articles={}
directory='/home/xxxiaoliu/qud-parsing/DCQA-Discourse-Comprehension-by-Question-Answering/article2/'
for filename in sorted(os.listdir(directory)):
    article = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(article):
        each_article=[]
        file=open(article,'r')
        for line in file:
            each_article.append(" ".join(line.strip().split(" ")[1:]))
        articles[filename[:4]] = each_article[:20]
        
data = []
with open('data/processed/single_joint_question_val_outputs.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))
        
reformat_pred = []
used_answer = dict()
for i in data:
    article_id = i['id'][:4]
    answer_id = i['meta']['answer_id'].zfill(2)
    anchor_id = i['meta']['pred_anchor_id'].zfill(2)
    
    if not article_id in used_answer:
        sentence_num = len(articles[article_id])
        used_answer[article_id] = []
        for j in range(sentence_num-1):
            used_answer[article_id].append([])
    if int(answer_id) > len(used_answer[article_id])+1:
        continue
    
    for j in i['output']:
        splits = re.split('answering the question of "(.*?)".', j)
        question = splits[1]
        used_answer[article_id][int(answer_id)-2].append({
            'anchor_id': anchor_id,
            'question': question
        })
        
        
for article_id in used_answer:
    for i in range(len(used_answer[article_id])):
        reformat_pred.append({
            'article_id': article_id,
            'answer_id': str(i+2).zfill(2),
            'candidates': used_answer[article_id][i]
        })

print(len(reformat_pred))
sorted_list = sorted(reformat_pred, key=lambda x: (x['article_id'], x['answer_id']))

with open('data/processed/reformat_single_joint_val.json', 'w') as f:
    f.write(json.dumps(sorted_list, indent=2)) 