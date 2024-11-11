import os
import copy
import json
import random
import csv
import re
import numpy as np
random.seed(0)

articles={}
directory='../../dcqa/article2/'
for filename in sorted(os.listdir(directory)):
    article = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(article):
        each_article=[]
        file=open(article,'r')
        for line in file:
            each_article.append(" ".join(line.strip().split(" ")[1:]))
        articles[filename[:4]] = each_article[:20]
        
def process_data(data, split):
    # transform the train/val data into instruction format
    prompt_format = "### Instruction:\n{instruction}\n\n### Input:\nContext: {context}\nAnswer sentence: {answer_sentence}\n\n### Response:"
    instruction = "Given the answer sentence, reason through the context to find the most likely sentence where a question can be generated."

    processed_data = []
    for k in data:
        article_id = k['Article'][0]['ArticleID'].zfill(4)
        current_article = articles[article_id]

        context = ""
        # add sentence id before each sentence
        for sentence in range(len(current_article)):
            context = context+' XT'+str(sentence+1).zfill(2)+' '+ current_article[sentence]  
        context = context[1:]

        used_answer_id = set()
        for idx, qas in enumerate(k['Article'][0]['qas']):
            answer_sentence_id = qas['AnswerSentenceID']
            if answer_sentence_id in used_answer_id or answer_sentence_id > 20:
                continue
            used_answer_id.add(answer_sentence_id)

            cur_context = context[:context.find('XT'+str(answer_sentence_id).zfill(2))].strip()
            answer_sentence = 'XT'+str(answer_sentence_id).zfill(2)+' '+ current_article[answer_sentence_id - 1]

            question_id = article_id + '_' + str(idx)

            anchor_sentence_id = qas['AnchorSentenceID']
            anchor_text = 'XT'+str(anchor_sentence_id).zfill(2)
            anchor_start = cur_context.find(anchor_text)

            question = qas['Question']
            answer = f'Sentence [{answer_sentence_id}] is anchored by sentence [{anchor_sentence_id}], answering the question of "{question}".'
            
            if anchor_start >= 0:
                if split == 'train':
                    processed_data.append({
                        'dataset': 'DCQA-single-joint-train',
                        'id': question_id,
                        'messages': [{"role": "user", "content": prompt_format.format(instruction = instruction, context = cur_context, answer_sentence = answer_sentence)}, {"role": "assistant", "content": answer}]
                    })
                else:
                    processed_data.append({
                        'context' : cur_context,
                        'answer' : answer_sentence,
                        'question' : question
                    })
            
            # if anchor_start >= 0:
            #     if split == 'train':
            #         processed_data.append({
            #             'dataset': 'DCQA-single-joint-train',
            #             'id': question_id,
            #             'messages': [{"role": "user", "content": prompt_format.format(instruction = instruction, context = cur_context, answer_sentence = answer_sentence)}, {"role": "assistant", "content": answer}]
            #         })
            #     else:
            #         processed_data.append({
            #             'dataset': 'DCQA-single-joint-val',
            #             'id': question_id,
            #             'prompt': prompt_format.format(instruction = instruction, context = cur_context, answer_sentence = answer_sentence),
            #             'reference': answer
            #         })

    print(len(data), len(processed_data))
    return processed_data
            
# train_data = json.load(open('../../dcqa/train.json', 'r'))
# processed_train_data = process_data(train_data, 'train')
# with open('data/processed/single_joint_train.jsonl', 'w') as f:
#     for i in processed_train_data:
#         f.write(json.dumps(i)+'\n')
        
val_data = json.load(open('../../dcqa/val.json', 'r'))
processed_val_data = process_data(val_data, 'val')
with open('processed/gpt4.jsonl', 'w') as f:
    for i in processed_val_data:
        f.write(json.dumps(i)+'\n')
        
# val_data = json.load(open('../../dcqa/test.json', 'r'))
# processed_val_data = process_data(val_data, 'val')
# with open('data/processed/single_joint_anchor_test.jsonl', 'w') as f:
#     for i in processed_val_data:
#         f.write(json.dumps(i)+'\n')