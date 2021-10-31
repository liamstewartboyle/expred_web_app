import pickle

import torch
import os

from debug import get_bogus_pred
from expred_utils import mark_evidence, merge_subtoken_exp_preds, pad_exp_pred, preprocess
from faxplain_utils import restore_from_temp, dump_quel
from search import google_rest_api_search as wiki_search
# from search import bing_wiki_search as wiki_search
# from search import google_wiki_search as wiki_search
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
temp_data_fname = f'data/temp_{session_id}.pkl'
temp_fname = f'data/temp_{session_id}.txt'
bert_dir = 'bert-base-uncased'
evi_finder_loc = './trained_models/fever/evidence_token_identifier.pt'
evi_finder_url = 'https://www.dropbox.com/s/qwinyap4kbxzdvn/evidence_token_identifier.pt?dl=1'
cls_loc = 'trained_models/fever/evidence_classifier.pt'
cls_url = 'https://www.dropbox.com/s/oc3qrgl0tqn9mqd/evidence_classifier.pt?dl=1'
classes = ["SUPPORTS", "REFUTES"]
device = torch.device('cpu')
default_ndocs = 3
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
                         use_half_precision=False,
                         load_from_ckpt=True)
    if not os.path.isfile(evi_finder_loc):
        import urllib
        urllib.request.urlretrieve(evi_finder_url, evi_finder_loc)
    evi_finder.load_state_dict(torch.load(evi_finder_loc, map_location=device), strict=False)

    cls = BertClassifier(bert_dir=bert_dir,
                         pad_token_id=tokenizer.pad_token_id,
                         cls_token_id=tokenizer.cls_token_id,
                         sep_token_id=tokenizer.sep_token_id,
                         num_labels=mtl_params.num_labels,
                         max_length=max_length,
                         mtl_params=mtl_params,
                         use_half_precision=False,
                         load_from_ckpt=True)
    if not os.path.isfile(cls_loc):
        import urllib
        urllib.request.urlretrieve(cls_url, cls_loc)
    cls.load_state_dict(torch.load(cls_loc, map_location=device), strict=False)

app = Flask(__name__)


def clean(query):
    query = query.strip().lower()
    return query


@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        query = request.form['query']
        query = clean(query)
        global default_ndocs
        default_ndocs = int(request.form['cod'])
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


@app.route('/prediction/<query>', methods=['GET', 'POST'])
def prediction(query):
    if request.method == 'POST':
        with open(temp_data_fname, 'rb') as fin:
            query, urls, docs, exps, labels = pickle.load(fin)

        disagree_idx = []
        for i in range(default_ndocs):
            if request.form[f'agree{i}'] == 'y':
                dump_quel(mgc_data_fname, query, urls[i:i + 1],
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
                aux_preds, exp_preds, att_masks = exp(queries, [i for i in range(default_ndocs)], docs)

                hard_exp_preds = torch.round(exp_preds)
                queries, docs = mark_evidence(queries, docs, hard_exp_preds, tokenizer, max_length)

                cls.eval()
                cls_preds = cls(queries, [i for i in range(default_ndocs)], docs)

                aux_preds = [classes[torch.argmax(p)] for p in aux_preds]
                cls_preds = [classes[torch.argmax(p)] for p in cls_preds]
                hard_exp_preds = [p[(len(queries[0]) + 2):] for p in hard_exp_preds]
                hard_exp_preds = merge_subtoken_exp_preds(hard_exp_preds, docs_slice)
            return aux_preds, cls_preds, hard_exp_preds#, docs_clean

        # wiki_urls = bing_wiki_search(query)[:default_ndocs]
        wiki_urls = wiki_search(query, default_ndocs)[:default_ndocs]
        if not wiki_urls:
            return render_template('no_results.html')
        orig_docs = [get_wiki_docs(url) for url in wiki_urls]
        (
            tokenized_query,
            query_slice,
            tokenized_docs,
            docs_slice,
            docs_split
        ) = preprocess(query, orig_docs, tokenizer, default_ndocs, max_sentence)
        queries = [torch.tensor(tokenized_query[0], dtype=torch.long) for i in range(default_ndocs)]
        docs = [torch.tensor(s, dtype=torch.long) for s in tokenized_docs]
        aux_preds, cls_preds, exp_preds = _predict(evi_finder, cls, queries, docs)
        exp_preds = [pad_exp_pred(exp, doc) for exp, doc in zip(exp_preds, docs_split)]
        pred = postprocess(cls_preds, exp_preds, docs_split, wiki_urls)
        pred['query'] = query
        pred['max_sentences'] = max_sentence
        with open(temp_data_fname, 'wb+') as fout:
            pickle.dump((query, wiki_urls, docs_split, exp_preds, cls_preds), fout)
    return render_template('predict.html', pred=pred)


@app.route('/select', methods=['GET', 'POST'])
def select():
    with open(temp_fname, 'r') as fin:
        disagree_idxs = eval(str(fin.read()))
    with open(temp_data_fname, 'rb') as fin:
        query, urls, docs, exps, labels = pickle.load(fin)
    input = [list(zip(urls, docs, exps, labels))[idx] for idx in disagree_idxs]
    urls, docs, exps, labels = zip(*input)

    # docs_trunc = [doc[:max_sentence] for doc in docs]
    # docs_rest = [doc[max_sentence] for doc in docs]

    if request.method == 'POST':
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

        evis, labels = _get_ugc(docs, request.form)
        dump_quel(ugc_data_fname, query, urls, docs, evis, labels)
        return render_template('thank_contrib.html')
    return render_template('select.html', query=query, docs=docs,
                           urls=urls, exps=exps, labels=labels, max_sentence=max_sentence)


if __name__ == "__main__":
    # app.run(host='127.0.0.1', port=8080)
    app.run(host='0.0.0.0', port=8080)