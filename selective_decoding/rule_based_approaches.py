import spacy 
import os 
import json
from tqdm import tqdm
import torch
from transformers import BartForSequenceClassification, BartTokenizer
from transformers import pipeline

#_______________________________________
candidate_file = 'single_joint_sample_val.json'
output_file = 'score_single_joint_val.json'
#________________________________________

def compute_givenness(question, context):
    
    doc_context = nlp(context)
    context_lemmas_list = [token.lemma_ for token in doc_context if token.pos_ in {'NOUN', 'VERB', 'ADJ', 'ADV'} ]

    doc_question = nlp(question)
    question_lemmas_list = [token.lemma_ for token in doc_question]
    new_lemmas=0
    for lemma in question_lemmas_list:
        if lemma not in context_lemmas_list:
            new_lemmas+=1
            
    score = new_lemmas/len(question_lemmas_list)
    return score

def compute_comp(question, answer):
    probs = classifier(question+answer, label_mapping)
    return float(probs['scores'][0])

def compute_relevance(question, anchor):
    
    doc = nlp(question)
    # print(question)
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    
    if len(noun_phrases) != 0:
        max_noun_phrase = max(noun_phrases, key=len)
        doc_np = nlp(max_noun_phrase)
        question_lemmas_list = [token.lemma_ for token in doc_np]
    else:
        question_lemmas_list = []
        
    #anchor sentence
    doc_anchor = nlp(anchor)
    anchor_lemmas_list = [token.lemma_ for token in doc_anchor if token.pos_ in {'NOUN', 'VERB', 'ADJ', 'ADV'}]
    
    new_lemmas=0
    for lemma in question_lemmas_list:
        if lemma in anchor_lemmas_list:
            new_lemmas+=1
    
    if len(question_lemmas_list) == 0:
        score = 0
    else:
        score = new_lemmas/len(question_lemmas_list)
    
    return score



with open(candidate_file, 'r') as f:
    val_data = json.load(f)
    

articles = {}
directory='../dcqa/article2/'
for filename in sorted(os.listdir(directory)):
    article = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(article):
        each_article=[]
        file=open(article,'r')
        for line in file:
            each_article.append(" ".join(line.strip().split(" ")[1:]))
        articles[filename[:4]] = each_article[:20]
    

nlp = spacy.load('en_core_web_sm')
classifier = pipeline("zero-shot-classification",model="facebook/bart-large-mnli")
label_mapping = ['valid', 'neutral', 'invalid']
        
        
for i in tqdm(val_data):
    article_id = i['article_id']
    answer = articles[article_id][int(i['answer_id'])-1]
    for j in i['candidates']:
        anchor_id = int(j['anchor_id'])
        context = ' '.join(articles[article_id][:anchor_id-1])
        anchor = articles[article_id][anchor_id-1]
        question = j['question']
        j['relv'] = compute_relevance(question, anchor)
        j['givn'] = compute_givenness(question, context)
        j['comp'] = compute_comp(question, answer)
        j['score'] = j['relv'] - j['givn']+j['comp']
    
with open(output_file, 'w') as f:
    f.write(json.dumps(val_data, indent=2)) 
