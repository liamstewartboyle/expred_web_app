#!/usr/bin/env python
# coding: utf-8

# In[2]:


import random


# In[3]:


import torch
from tokenizer import BertTokenizerWithMapping
from models.mlp import BertMTL, BertClassifier
from models.params import MTLParams
from copy import deepcopy
import os 
from flask import Flask, request, redirect, url_for, render_template
from azure_search.bing_utils import bing_wiki_search, get_wiki_docs
from itertools import chain
import numpy as np 
import csv
import random

session_id = hex(int(random.random()*1e13))[2:]

import os
if os.path.isdir('data'):
    print('data folder exists')
else:
    print('creating data folder')
    os.mkdir('data')

ugc_data_fname = f'data/ugc_{session_id}.csv' # user generated content
mgc_data_fname = f'data/mgc_{session_id}.csv' # machine genarated content
temp_data_fname = f'data/temp_{session_id}.csv'
temp_fname = f'data/temp_{session_id}.txt'
bert_dir = 'bert-base-uncased'
evi_finder_loc = './trained_models/fever/evidence_token_identifier.pt'
cls_loc = 'trained_models/fever/evidence_classifier.pt'
classes = ["SUPPORTS", "REFUTES"]
device = torch.device('cpu')
top = 3
max_sentence = 30
debug = True
debug = False


# In[4]:


if debug:
    print('debug, will not load models')
else:
    print("Loading models")
    tokenizer = BertTokenizerWithMapping.from_pretrained(bert_dir)
    max_length = 512
    use_half_precision = False

    mtl_params = MTLParams(dim_cls_linear=256, num_labels=2, dim_exp_gru=128)

    evi_finder = BertMTL(bert_dir=bert_dir,
                         tokenizer=tokenizer,
                         mtl_params=mtl_params,
                         use_half_precision=False)
    evi_finder.load_state_dict(torch.load(evi_finder_loc, map_location=device))

    cls = BertClassifier(bert_dir=bert_dir,
                         pad_token_id=tokenizer.pad_token_id,
                         cls_token_id=tokenizer.cls_token_id,
                         sep_token_id=tokenizer.sep_token_id,
                         num_labels=mtl_params.num_labels,
                         max_length=max_length,
                         mtl_params=mtl_params,
                         use_half_precision=False)
    cls.load_state_dict(torch.load(cls_loc, map_location=device))


# In[5]:


app = Flask(__name__)


# In[6]:


def clean(query):
    query = query.strip().lower()
    return query


@app.route('/', methods=['GET', 'POST']) 
def main_page():
    if request.method == 'POST':
        query = request.form['query']
        query = clean(query)
        return redirect(url_for('prediction', query=query))
    return render_template('index.html')


def preprocess(query, docs):
    query = query.split()
    tokenized_q, tokenized_q_token_slides = tokenizer.encode_docs([[query]])
    tokenized_q = tokenized_q[0]
    tokenized_q_token_slide = tokenized_q_token_slides[0]
    docs_clean = [[list(chain.from_iterable([s.split() + ['.'] for s in d.lower().split('.')][:max_sentence]))] 
                  for d in docs]
    docs = deepcopy(docs_clean)
    tokenized_docs, tokenized_doc_token_slides = tokenizer.encode_docs(docs)
    tokenized_docs = [list(chain.from_iterable(tokenized_docs[i])) for i in range(top)]
    tokenized_doc_token_slides = [list(chain.from_iterable(tokenized_doc_token_slides[i]))
                                  for i in range(top)]
    return tokenized_q, tokenized_q_token_slide,           tokenized_docs, tokenized_doc_token_slides,           docs_clean


def mark_evidence(queries, docs, hard_preds, wildcard='.'):
    wildcard_tensor = tokenizer.convert_tokens_to_ids('.') * torch.ones(max_length).type(torch.int)
    doc_max_len = max_length - 2 - len(queries[0])
    docs = [d[:doc_max_len] if len(d) >= doc_max_len 
                            else torch.cat([d, torch.zeros(doc_max_len-len(d)).type(torch.int64)])
            for d in docs]
    new_docs = []
    for q, d, e in zip(queries, docs, hard_preds):
        temp = torch.cat([torch.zeros(1).type(torch.int64), q, torch.zeros(1).type(torch.int64), d])
        temp = e * temp + (1-e) * wildcard_tensor
        new_docs.append(temp[(len(queries[0]) + 2):].type(torch.int64))
    return queries, new_docs


def adapt_exp_pred(exp, doc):
    exp = exp[:len(doc[0])] + [0] * (len(doc[0]) - len(exp))
    return exp


def merge_subtoken_exp_preds(exp_preds, slides):
    ret = []
    for p, ss in zip(exp_preds, slides):
        p = p.tolist()
        ret.append([max(p[s[0]:s[1]] + [0]) for s in ss])
    return ret


def color_cls_pred(c, 
                   pos_label='SUPPORTS', pos_color='green', 
                   neg_label='REFUTES', neg_color='red',
                   default_color='gray'):
    color = default_color
    if c == pos_label:
        color = pos_color
    elif c == neg_label:
        color = neg_color
    return f'<p style="color:{color};">{c}</p>'

    
def highlight_exp_pred(exp, doc):
    ret = ''
    abrcount = 0
    abrflag = False  # for abbreviation
    for e, w in zip(exp, doc[0]):
        if e == 1:
            ret += f'<span style="background-color:tomato; float: left">{w}&nbsp;</span>'
            abrcount = 0
            abrflag = False
        else:
            if abrflag:
                continue
            abrcount += 1
            if abrcount > 4:
                abrflag = True
                ret += f'<span style="float:left">...&nbsp;</span>'
            else:
                ret += f'<span style="float:left">{w}&nbsp;</span>'
    return ret


def postprocess(cls_preds, exp_preds, docs_clean, urls):
    cls_strs = [color_cls_pred(c) for c in cls_preds]
    evi_strs = [highlight_exp_pred(exp, doc) for exp, doc in zip(exp_preds, docs_clean)]
    pred = {
        'clses': cls_strs,
        'evis': evi_strs,
        'links': urls
    }        
    return pred
    

def get_bogus_pred():
    pred = {
        'clses': ['REFUTE' for i in range(3)],
        'evis': [[1 for j in range(3)] for i in range(3)],
        'links': ['www.abs.com' for i in range(3)],
        'query': 'this is not a query'
    }
    return pred


if not os.path.isfile(ugc_data_fname):
    with open(ugc_data_fname, 'w+', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow('query url evidence label'.split())
    
if not os.path.isfile(mgc_data_fname):
    with open(mgc_data_fname, 'w+', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow('query url evidence label'.split())
        
        
def dump_quel(fname, query, urls, docs, exps, labels, mode='a+'):
    with open(fname, mode, newline='') as fout:
        for url, doc, exp, label in zip(urls, docs, exps, labels):
#             print(doc, exp)
            assert len(doc[0]) == len(exp)
            writer = csv.writer(fout)
            writer.writerow([query, url, list(zip(doc[0], exp)), label])
            

def restore_from_temp(temp_fname, idxs=None):
    query = None
    urls, docs, exps, labels = [], [], [], []
    with open(temp_fname, 'r', newline='') as fin:
        reader = csv.reader(fin)
        for idx, (query, url, evidence, label) in enumerate(reader):
            if idxs is None or idx in idxs:
                evidence = eval(evidence)
                doc, exp = zip(*evidence)
                urls.append(url)
                docs.append([doc])
                exps.append(exp)
                labels.append(label)
    return query, urls, docs, exps, labels


@app.route('/select', methods=['GET', 'POST'])
def select():
    with open(temp_fname, 'r') as fin:
        disagree_idxs = eval(str(fin.read()))
    query, urls, docs, exps, labels = restore_from_temp(temp_data_fname, idxs=disagree_idxs)
    def _get_ugc(docs, form):
        labels = []
#         print([i for i in form.keys()])
        evis = [[0 for w in doc[0]] for doc in docs]
        bad_docs = []            
        for i in range(len(evis)):
            labels.append(form[f"cls{i}"])
            if labels[-1] == 'BAD_DOC':
                continue
#             print([form.get(f'exp{i},{j}') for j in range(3)])
#             print(len(docs[i]))
            for j in range(len(docs[i][0])):
#                 print(evis, labels)
                if form.get(f'exp{i},{j}') is not None:
                    evis[i][j] = 1
#         print(evis, labels)    
        return evis, labels       
    if request.method == 'POST':
        evis, labels = _get_ugc(docs, request.form)
        dump_quel(ugc_data_fname, query, urls, docs, evis, labels)
        return render_template('thank_contrib.html')
#     print(docs)
    return render_template('select.html', query=query, docs=docs, 
                           urls=urls, exps=exps, labels=labels)

                                        
@app.route('/prediction/<query>', methods=['GET', 'POST'])
def prediction(query):
    if request.method == 'POST':
#         print(request.form['satisfy'])
        query, urls, docs, exps, labels = restore_from_temp(temp_data_fname)
        disagree_idx = []
        for i in range(top):
            if request.form[f'agree{i}'] == 'y':
                dump_quel(mgc_data_fname, query[i:i+1], urls[i:i+1],
                          docs[i:i+1], exps[i:i+1], labels[i:i+1])
            else:
                disagree_idx.append(i)
        if len(disagree_idx) == 0:
            return render_template('thankyou.html')
        else:
            with open(temp_fname, 'w+') as fout:
                fout.write(str(disagree_idx))
            return redirect(url_for('select'))
#             with open(mgc_data_fname, 'a+') as fout:
#                 for url, doc, exp, label in zip(urls, docs, exps, labels):
#                     fout.write(f"{query}, {url}, {list(zip(doc[0], exp))}, {label}\n")
#             return render_template('thankyou.html')
#         if request.form['satisfy'] == 'No...':
# #             print(url_for('select', query, docs))
#             return redirect(url_for('select'))
    if debug:
        pred = get_bogus_pred()
        query = pred['query']
        docs_clean = [[['a' for a in range(3)]] for b in range(3) ]
        exp_preds = [[1]*3 for i in range(3)]
        cls_preds = ['REFUTES' for i in range(3)]
        wiki_urls = pred['links']
    else:
        def _predict(exp, cls, queries, docs):
            with torch.no_grad():
                exp.eval()

                aux_preds, exp_preds, att_masks = exp(queries, [i for i in range(top)], docs)

                hard_exp_preds = torch.round(exp_preds)
                queries, docs = mark_evidence(queries, docs, hard_exp_preds)

                cls_preds = cls(queries, [i for i in range(top)], docs)

                aux_preds = [classes[torch.argmax(p)] for p in aux_preds]
                cls_preds = [classes[torch.argmax(p)] for p in cls_preds]
                hard_exp_preds = [p[(len(queries[0]) + 2):] for p in hard_exp_preds]
                hard_exp_preds = merge_subtoken_exp_preds(hard_exp_preds, tokenized_d_token_slides)
            return aux_preds, cls_preds, hard_exp_preds, docs_clean

        wiki_urls = bing_wiki_search(query)[:top]
        orig_docs = [get_wiki_docs(url) for url in wiki_urls]
        tokenized_q, tokenized_q_token_slide, tokenized_ds, tokenized_d_token_slides, docs_clean =             preprocess(query, orig_docs)
        queries = [torch.tensor(tokenized_q[0], dtype=torch.long) for i in range(top)]
        docs = [torch.tensor(s, dtype=torch.long) for s in tokenized_ds]
        aux_preds, cls_preds, exp_preds, docs_clean = _predict(evi_finder, cls, queries, docs)
        exp_preds = [adapt_exp_pred(exp, doc) for exp, doc in zip(exp_preds, docs_clean)]
        pred = postprocess(cls_preds, exp_preds, docs_clean, wiki_urls)
        pred['query'] = query
    dump_quel(temp_data_fname, query, wiki_urls, docs_clean, exp_preds, cls_preds, 'w+')
    return render_template('predict.html', pred=pred)


# In[7]:

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080)

