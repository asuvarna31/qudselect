import os
import copy
import json
import random
import csv
import re
import numpy as np
random.seed(0)
        
def process_question_data(data):
    # construct question generation data based on predicted anchors 
    prompt_format_a = "\nSentence [{answer_id}] is anchored by sentence [{anchor_id}],"

    processed_data = []
    anchor_num = []
    for i in pred_data:
        anchors = []
        for j in i['output']:
            splits = re.split('Sentence \[(\d+)\] is anchored by sentence \[(\d+)\],', j)
            if len(splits) > 2:
                answer_id = splits[1]
                pred = splits[2]
                if not pred in anchors:
                    anchors.append(pred)
        anchor_num.append(len(anchors))
        for idxj, j in enumerate(anchors):
            if int(answer_id) > int(j) and int(j) > 0:
                processed_data.append({
                    'dataset': 'DCQA-single-joint-question-val',
                    'id': i['id']+'_'+str(idxj),
                    'prompt': i['prompt'] + prompt_format_a.format(answer_id = answer_id, anchor_id = j),
                    'reference': i['reference'],
                    'meta': {'answer_id': answer_id, 'pred_anchor_id': j}
                })
    print(len(pred_data), len(processed_data))
    print(np.mean(anchor_num))
    
    return processed_data
            
pred_data = []
with open('data/processed/single_joint_anchor_val_outputs.jsonl', 'r') as f:
    for line in f:
        pred_data.append(json.loads(line))

processed_question_data = process_question_data(pred_data)
with open('data/processed/single_joint_question_val.jsonl', 'w') as f:
    for i in processed_question_data:
        f.write(json.dumps(i)+'\n')