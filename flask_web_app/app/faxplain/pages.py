import os
import pickle

import torch
from flask import redirect, render_template, request, url_for, Blueprint

from utils.debug import get_bogus_pred
from faxplain.utils import machine_rationale_mask_to_html, dump_quel
from faxplain.search import google_rest_api_search as wiki_search
from faxplain.wiki import get_wiki_docs
from utils.logger import get_logger
from preprocess import clean

debug = False

faxplain_page = Blueprint('faxplain', __name__, template_folder='templates')

FLASK_ENV = os.getenv('FLASK_ENV')
development = True if (FLASK_ENV is not None and 'development' in FLASK_ENV) else False

logger = get_logger('faxplain')


@faxplain_page.route('/faxplain', methods=['GET', 'POST'])
def faxplain():
    if request.method == 'POST':
        query = request.form['query']
        query = clean(query)
        global default_ndocs
        default_ndocs = int(request.form['cod'])
        return redirect(url_for('prediction', query=query))
    return render_template('index.html')


@faxplain_page.route('/prediction/<query>', methods=['GET', 'POST'])
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
        # print(pred, pred.__dir__(), pred['query'], pred['query'].__dir__())
        docs_clean = [[['a' for a in range(3)]] for b in range(3)]
        exp_preds = [[1] * 3 for i in range(3)]
        cls_preds = ['REFUTES' for i in range(3)]
        wiki_urls = pred['links']
    else:
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
        ) = preprocess(query, orig_docs, span_tokenizer, basic_tokenizer, default_ndocs, max_sentence)
        queries = [torch.tensor(tokenized_query[0], dtype=torch.long)
                   for i in range(default_ndocs)]
        docs = [torch.tensor(s, dtype=torch.long) for s in tokenized_docs]
        aux_preds, cls_preds, exp_preds, masked_docs = expred_predict(
            mtl_module, cls_module, queries, docs, docs_slice, span_tokenizer)
        exp_preds = merge_subtoken_exp_preds(exp_preds, docs_slice)
        exp_preds = [pad_mask_to_doclen(exp, doc)
                     for exp, doc in zip(exp_preds, docs_split)]
        pred = machine_rationale_mask_to_html(
            cls_preds, exp_preds, docs_split, wiki_urls)
        pred['query'] = query
        pred['max_sentences'] = max_sentence
        with open(temp_data_fname, 'wb+') as fout:
            pickle.dump((query, wiki_urls, docs_split,
                         exp_preds, cls_preds), fout)
    return render_template('predict.html', pred=pred)


@faxplain_page.route('/select', methods=['GET', 'POST'])
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
                labels.append(form[f"cls_module{i}"])
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
