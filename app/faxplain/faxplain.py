import torch
import os

from debug import get_bogus_pred
from expred_utils import mark_evidence, merge_subtoken_exp_preds, adapt_exp_pred, preprocess
from faxplain_utils import restore_from_temp, dump_quel
from google_search import google_wiki_search
from tokenizer import BertTokenizerWithMapping
from models.mlp import BertMTL, BertClassifier
from models.params import MTLParams
from flask import Flask, request, redirect, url_for, render_template
import csv
import random

from utils import color_cls_pred, highlight_exp_pred
from wiki import get_wiki_docs

session_id = hex(int(random.random() * 1e13))[2:]

if os.path.isdir('data'):
    print('data folder exists')
else:
    print('creating data folder')
    os.mkdir('data')

ugc_data_fname = f'data/ugc_{session_id}.csv'  # user generated content
mgc_data_fname = f'data/mgc_{session_id}.csv'  # machine genarated content
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

app = Flask(__name__)


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


def postprocess(cls_preds, exp_preds, docs_clean, urls):
    cls_strs = [color_cls_pred(c) for c in cls_preds]
    evi_strs = [highlight_exp_pred(exp, doc) for exp, doc in zip(exp_preds, docs_clean)]
    urls = [url.split('/')[-1] for url in urls]
    pred = {
        'clses': cls_strs,
        'evis': evi_strs,
        'links': urls
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


@app.route('/select', methods=['GET', 'POST'])
def select():
    with open(temp_fname, 'r') as fin:
        disagree_idxs = eval(str(fin.read()))
    query, urls, docs, exps, labels = restore_from_temp(temp_data_fname, idxs=disagree_idxs)

    def _get_ugc(docs, form):
        labels = []
        evis = [[0 for w in doc[0]] for doc in docs]
        for i in range(len(docs)):
            labels.append(form[f"cls{i}"])
            if labels[-1] == 'BAD_DOC':
                continue
            for j in range(len(docs[i][0])):
                if form.get(f'exp{i},{j}') is not None:
                    evis[i][j] = 1
        return evis, labels

    if request.method == 'POST':
        evis, labels = _get_ugc(docs, request.form)
        dump_quel(ugc_data_fname, query, urls, docs, evis, labels)
        return render_template('thank_contrib.html')
    return render_template('select.html', query=query, docs=docs,
                           urls=urls, exps=exps, labels=labels)


@app.route('/prediction/<query>', methods=['GET', 'POST'])
def prediction(query):
    if request.method == 'POST':
        query, urls, docs, exps, labels = restore_from_temp(temp_data_fname)
        disagree_idx = []
        for i in range(top):
            if request.form[f'agree{i}'] == 'y':
                dump_quel(mgc_data_fname, query[i:i + 1], urls[i:i + 1],
                          docs[i:i + 1], exps[i:i + 1], labels[i:i + 1])
            else:
                disagree_idx.append(i)
        if len(disagree_idx) == 0:
            return render_template('thankyou.html')
        else:
            with open(temp_fname, 'w+') as fout:
                fout.write(str(disagree_idx))
            return redirect(url_for('select'))
    if debug:
        pred = get_bogus_pred()
        query = pred['query']
        print(pred, pred.__dir__(), pred['query'], pred['query'].__dir__())
        docs_clean = [[['a' for a in range(3)]] for b in range(3)]
        exp_preds = [[1] * 3 for i in range(3)]
        cls_preds = ['REFUTES' for i in range(3)]
        wiki_urls = pred['links']
    else:
        def _predict(exp, cls, queries, docs):
            with torch.no_grad():
                exp.eval()

                aux_preds, exp_preds, att_masks = exp(queries, [i for i in range(top)], docs)

                hard_exp_preds = torch.round(exp_preds)
                queries, docs = mark_evidence(queries, docs, hard_exp_preds, tokenizer, max_length)

                cls_preds = cls(queries, [i for i in range(top)], docs)

                aux_preds = [classes[torch.argmax(p)] for p in aux_preds]
                cls_preds = [classes[torch.argmax(p)] for p in cls_preds]
                hard_exp_preds = [p[(len(queries[0]) + 2):] for p in hard_exp_preds]
                hard_exp_preds = merge_subtoken_exp_preds(hard_exp_preds, tokenized_d_token_slides)
            return aux_preds, cls_preds, hard_exp_preds, docs_clean

        # wiki_urls = bing_wiki_search(query)[:top]
        wiki_urls = google_wiki_search(query, top)[:top]
        orig_docs = [get_wiki_docs(url) for url in wiki_urls]
        (
            tokenized_q,
            tokenized_q_token_slide,
            tokenized_ds,
            tokenized_d_token_slides,
            docs_clean
        ) = preprocess(query, orig_docs, tokenizer, top, max_sentence)
        queries = [torch.tensor(tokenized_q[0], dtype=torch.long) for i in range(top)]
        docs = [torch.tensor(s, dtype=torch.long) for s in tokenized_ds]
        aux_preds, cls_preds, exp_preds, docs_clean = _predict(evi_finder, cls, queries, docs)
        exp_preds = [adapt_exp_pred(exp, doc) for exp, doc in zip(exp_preds, docs_clean)]
        pred = postprocess(cls_preds, exp_preds, docs_clean, wiki_urls)
        pred['query'] = query
    dump_quel(temp_data_fname, query, wiki_urls, docs_clean, exp_preds, cls_preds, 'w+')
    return render_template('predict.html', pred=pred)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080)
