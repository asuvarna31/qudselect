import json
from tqdm import tqdm

with open('scored_single_joint_val.json', 'r') as f:
    val_data = json.load(f)
    

new_list = []
for i in tqdm(val_data):
    new = {}
    new['article_id'] = i['article_id']
    new['answer'] = i['answer_id']
    if len(i['candidates']):
        maxscore = max(i['candidates'], key=lambda x:x['score']) 
        new['score'] = maxscore['score']
        new['anchor_id'] = maxscore['anchor_id']
        new['question'] = maxscore['question']
        new_list.append(new)
            
with open('final_quds.json', 'w') as f:
    f.write(json.dumps(new_list, indent=2)) 

        