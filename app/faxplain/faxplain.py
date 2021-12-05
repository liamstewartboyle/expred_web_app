import pickle
from itertools import chain

import torch
import os

from debug import get_bogus_pred
from expred import expred
from expred.expred.eraser_utils import get_docids_seq, load_eraser_data, load_documents
from expred.expred.models.utils import fix_name_bare_bert
from expred_utils import mark_evidence, merge_subtoken_exp_preds, pad_exp_pred, preprocess
from faxplain_utils import restore_from_temp, dump_quel
from search import google_rest_api_search as wiki_search
# from search import bing_wiki_search as wiki_search
# from search import google_wiki_search as wiki_search
from tokenizer import BertTokenizerWithMapping
from transformers import BasicTokenizer
from expred.expred.models.mlp_mtl import BertMTL, BertClassifierHotflip
from models.params import MTLParams
from flask import Flask, request, redirect, url_for, render_template
from expred.expred.apply_hotflip import hotflip
import csv
import random

from utils import color_cls_pred, highlight_exp_pred
from wiki import get_wiki_docs

app = Flask(__name__)

application = 'counterfactual'

dataset_dir = os.environ.get('HOME') + '/.keras/datasets/movies'
if application == 'faxplain':
    evi_finder_loc = './trained_models/fever/evidence_token_identifier.pt'
    evi_finder_url = 'https://www.dropbox.com/s/qwinyap4kbxzdvn/evidence_token_identifier.pt?dl=1'
    cls_loc = 'trained_models/fever/evidence_classifier.pt'
    cls_url = 'https://www.dropbox.com/s/oc3qrgl0tqn9mqd/evidence_classifier.pt?dl=1'
    class_names = ["SUPPORTS", "REFUTES"]
elif application == 'counterfactual':
    evi_finder_loc = './trained_models/movies/evidence_token_identifier.pt'
    evi_finder_url = 'https://www.dropbox.com/s/qen0vx2uz6ksn3m/evidence_token_identifier.pt?dl=1'
    cls_loc = 'trained_models/movies/evidence_classifier.pt'
    cls_url = 'https://www.dropbox.com/s/0sfrdykcg6cf6kh/evidence_classifier.pt?dl=1'
    class_names = ["NEG", "POS"]
    data = {x.ann_id: x for x in chain.from_iterable(load_eraser_data(dataset_dir, merge_evidences=True))}
    docs = load_documents(dataset_dir)


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


def machine_rationale_mask_to_html(cls_preds, exp_preds, docs_clean, urls):
    cls_strs = [color_cls_pred(c) for c in cls_preds]
    evi_strs = [highlight_exp_pred(exp, doc) for exp, doc in zip(exp_preds, docs_clean)]
    urls = [url.split('/')[-1] for url in urls]
    pred = {
        'clses': cls_strs,
        'evis': evi_strs,
        'links': urls
    }
    return pred


def expred_inputs_preprocess(queries, masked_docs, max_length=512):
    cls_token = torch.LongTensor([tokenizer.cls_token_id])
    sep_token = torch.LongTensor([tokenizer.sep_token_id])
    # print(queries, masked_docs)
    inputs = []
    attention_masks = []
    for query, mdoc in zip(queries, masked_docs):
        d = torch.cat((cls_token, query, sep_token, mdoc), dim=-1)
        if len(d) > max_length:
            inputs.append(d[:max_length])
            attention_masks.append(torch.ones(max_length).type(torch.float))
        else:
            pad = torch.zeros(max_length - len(d))
            inputs.append(torch.cat((d, pad)))
            attention_masks.append(torch.cat((torch.ones_like(d), pad)).type(torch.float))
    if isinstance(inputs, list):
        return torch.vstack(inputs).type(torch.long), torch.vstack(attention_masks)
    return torch.LongTensor(inputs), torch.FloatTensor(attention_masks)


def exp_pred(exp_module:expred.models.mlp_mtl.BertMTL, queries, docs):
    with torch.no_grad():
        exp_module.eval()
        max_length = min(max(map(len, docs)) + 2 + len(queries[0]), 512)
        exp_inputs, attention_masks = expred_inputs_preprocess(queries, docs, max_length=max_length)
        aux_preds, exp_preds = exp_module(inputs=exp_inputs, attention_masks=attention_masks)
    aux_preds = [class_names[torch.argmax(p)] for p in aux_preds]

    hard_exp_preds = torch.round(exp_preds)
    masked_docs = mark_evidence(queries, docs, hard_exp_preds, tokenizer, max_length)
    docs_start = len(queries[0]) + 2
    docs_end = [docs_start + len(d) for d in masked_docs]
    hard_exp_preds = [p[docs_start: doc_end] for p, doc_end in zip(hard_exp_preds, docs_end)]

    return aux_preds, hard_exp_preds, masked_docs


def cls_pred(cls_module:expred.models.mlp_mtl.BertClassifierHotflip, queries, masked_docs):
    with torch.no_grad():
        cls_module.eval()
        cls_inputs, attention_masks = expred_inputs_preprocess(queries, masked_docs)
        cls_preds, _ = cls_module(inputs=cls_inputs, attention_masks=attention_masks)

    cls_preds = [class_names[torch.argmax(p)] for p in cls_preds]
    return cls_preds


def expred_predict(mtl_module, cls_module, queries, docs, docs_slice):
    aux_preds, hard_exp_preds, masked_docs = exp_pred(mtl_module, queries, docs)
    cls_preds = cls_pred(cls_module, queries, masked_docs)
    return aux_preds, cls_preds, hard_exp_preds#, docs_clean


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
        ) = preprocess(query, orig_docs, tokenizer, basic_tokenizer, default_ndocs, max_sentence)
        queries = [torch.tensor(tokenized_query[0], dtype=torch.long) for i in range(default_ndocs)]
        docs = [torch.tensor(s, dtype=torch.long) for s in tokenized_docs]
        aux_preds, cls_preds, exp_preds = expred_predict(mtl_module, cls_module, queries, docs, docs_slice)
        exp_preds = merge_subtoken_exp_preds(exp_preds, docs_slice)
        exp_preds = [pad_exp_pred(exp, doc) for exp, doc in zip(exp_preds, docs_split)]
        pred = machine_rationale_mask_to_html(cls_preds, exp_preds, docs_split, wiki_urls)
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


# def counterfactual_get():
#     query = request.args.get('query')
#     orig_doc = request.args.get('doc')
#     (
#         tokenized_query,
#         query_slice,
#         tokenized_docs,
#         docs_slice,
#         docs_split
#     ) = preprocess(query, orig_docs, tokenizer, 1, max_sentence)
#     # [0] for there is only one query and only 1 doc for the counterfactual generation
#     queries = [torch.tensor(tokenized_query[0], dtype=torch.long)]
#     docs = [torch.tensor(tokenized_docs[0], dtype=torch.long)]
#     cls_preds = cls_pred(cls_module, queries, docs)
#     docs_history = orig_doc
#     for i in range(n_word_replace):
#
#     return render_template('counterfactual.html',
#                            doc=docs[0],
#                            cls_pred=cls_preds[0],
#                            patch=)


def mask_irrational_tokens(docs, masks, wild_card='.'):
    wild_card_id = tokenizer.convert_tokens_to_ids([wild_card])[0]
    masks = [torch.Tensor(m) for m in masks]
    masked_docs = [d * m + wild_card_id * (1 - m) for d, m in zip(docs, masks)]
    return masked_docs


def get_hotflip(orig_query, orig_doc, cls_label, top_poses):
    (
        encoded_queries,
        query_slice,
        encoded_docs,
        docs_slice,
        docs_split
    ) = preprocess(orig_query, [orig_doc], tokenizer, basic_tokenizer, 1, max_sentence)
    encoded_queries = [torch.tensor(encoded_queries[0], dtype=torch.long)]
    encoded_docs = [torch.tensor(encoded_docs[0], dtype=torch.long)]
    encoded_query = encoded_queries[0]
    encoded_doc = encoded_docs[0]
    # the exp_preds are on docs only (no queries or overhead), thus len(exp_preds[0]) == len(tokenized_docs[0].shape)
    aux_preds, cls_preds, exp_preds = expred_predict(mtl_module, cls_module, encoded_queries, encoded_docs, docs_slice)
    exp_preds = [pad_exp_pred(exp, doc) for exp, doc in zip(exp_preds, encoded_docs)]
    mtl_masked_doc = mask_irrational_tokens(encoded_docs, exp_preds)
    mtl_masked_doc = ' '.join(tokenizer.convert_ids_to_tokens(mtl_masked_doc[0].tolist()))

    cls_input = torch.cat(
        (torch.Tensor([tokenizer.cls_token_id]),
         encoded_query,
         torch.Tensor([tokenizer.sep_token_id]),
         encoded_doc))
    cls_input = torch.Tensor(cls_input).type(torch.long)
    # mtl_masked_data = ['[CLS]'] + basic_tokenizer.tokenize(orig_query) + ['[SEP]'] + basic_tokenizer.tokenize(orig_doc)
    # hard_ann_masks = torch.cat((torch.zeros(len(encoded_query) + 2),
    #                             torch.ones(len(encoded_doc))), dim=-1)
    # print('cls_input: ', cls_input)
    # print('mtl_masked_sentences: ', mtl_masked_doc)
    # print('exp_preds: ', exp_preds)
    cls_attention_masks = torch.ones_like(cls_input).type(torch.float)

    top_poses = min(top_poses, len(encoded_doc))

    hotflip_res = hotflip(cls_module, tokenizer,
                          cls_input.unsqueeze(0), [[mtl_masked_doc]], exp_preds, cls_attention_masks.unsqueeze(0),
                          cls_label.unsqueeze(0),
                          label_classes=class_names,
                          top_poses=top_poses,
                          n_word_flip=n_word_replace,
                          device='cpu')
    hotflip_res[0]['input'] = tokenizer.convert_ids_to_tokens(encoded_doc)
    hotflip_res[0]['mask'] = exp_preds[0]
    # print(hotflip_res)
    return hotflip_res


def random_select_data(data, docs):
    # k = random.choice(list(data.keys()))
    k = 'negR_260.txt'
    selected_ann = data[k]
    print("key: ", k)
    ann_id = selected_ann.ann_id
    docid = selected_ann.docid
    doc = docs[docid]
    query = selected_ann.query
    return ann_id, query, doc


@app.route('/counterfactual', methods=['GET', 'POST'])
def counterfactual():
    cls_module.return_cls_embedding = False
    cls_module.return_bert_embedding = True
    annotation_id, orig_query, orig_sentences = random_select_data(data, docs)
    # cls_label = torch.LongTensor([1])
    # hotflip_res = get_hotflip(orig_query, orig_doc, cls_label, top_pos)
    orig_doc = ' '.join(chain.from_iterable(orig_sentences))
    label = data[annotation_id].label
    (
        tokenized_query,
        query_slice,
        tokenized_docs,
        docs_slice,
        docs_split
    ) = preprocess(orig_query, [orig_doc], tokenizer, basic_tokenizer, 1, 30)
    id_queries = [torch.tensor(tokenized_query[0], dtype=torch.long) for i in range(default_ndocs)]
    id_docs = [torch.tensor(s, dtype=torch.long) for s in tokenized_docs]
    aux_preds, cls_preds, exp_preds = expred_predict(mtl_module, cls_module, id_queries, id_docs, docs_slice)
    exp_preds = merge_subtoken_exp_preds(exp_preds, docs_slice)
    orig_doc = []
    exp_start = 0
    for sent_tokens in orig_sentences:
        orig_doc.append(highlight_exp_pred(exp_preds[0][exp_start: exp_start + len(sent_tokens)], [sent_tokens],
                                           highlight='bold', shorten=False))
        exp_start += len(sent_tokens)

    return render_template('counterfactual.html',
                           query=orig_query,
                           orig_doc=orig_doc,
                           label=label,
                           pred=cls_preds[0])


@app.route('/show_example', methods=['GET'])
def show_example():
    orig_query = request.args.get('query').strip()
    orig_doc = request.args.get('doc').strip()
    orig_label = request.args.get('label')
    # print(len(orig_query), orig_query)
    # print(len(orig_doc), orig_doc)
    # print(orig_label)
    # (
    #     tokenized_query,
    #     query_slice,
    #     tokenized_docs,
    #     docs_slice,
    #     docs_split
    # ) = preprocess(orig_query, [orig_doc], tokenizer, basic_tokenizer, 1, max_sentence)
    # [0] for there is only one query and only 1 doc for the counterfactual generation
    # queries = [torch.tensor(tokenized_query[0], dtype=torch.long)]
    # docs = [torch.tensor(tokenized_docs[0], dtype=torch.long)]
    cls_label = torch.LongTensor(class_names.index(orig_label))
    hotflip_res = get_hotflip(orig_query, orig_doc, cls_label, top_pos)
    return render_template('show_example.html', hotflip_res=hotflip_res)


@app.route('/doc-history', methods=['GET'])
def query_history():
    orig_query = request.args.get('query')
    orig_doc = request.args.get('doc')
    (
        tokenized_query,
        query_slice,
        tokenized_docs,
        docs_slice,
        docs_split
    ) = preprocess(orig_query, [orig_doc], tokenizer, basic_tokenizer, 1, max_sentence)
    # [0] for there is only one query and only 1 doc for the counterfactual generation
    queries = [torch.tensor(tokenized_query[0], dtype=torch.long)]
    docs = [torch.tensor(tokenized_docs[0], dtype=torch.long)]
    cls_preds = cls_pred(cls_module, queries, docs)
    return render_template('doc-history.html', doc_history=[orig_doc], preds=cls_preds)


if __name__ == "__main__":
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

    device = torch.device('cpu')
    default_ndocs = 3
    max_sentence = 30
    n_word_replace = 5
    debug = True
    debug = False
    top_pos = 10

    if debug:
        print('debug, will not load models')
    else:
        print("Loading models")
        tokenizer = BertTokenizerWithMapping.from_pretrained(bert_dir)
        basic_tokenizer = BasicTokenizer()
        max_length = 512
        use_half_precision = False

        mtl_params = MTLParams(dim_cls_linear=256, num_labels=2, dim_exp_gru=128)

        mtl_module = BertMTL(bert_dir=bert_dir,
                             tokenizer=tokenizer,
                             mtl_params=mtl_params,
                             use_half_precision=False,
                             load_from_local_ckpt=True)
        if not os.path.isfile(evi_finder_loc):
            os.makedirs(evi_finder_loc[::-1].split('/', 1)[1][::-1], exist_ok=True)
            import urllib

            urllib.request.urlretrieve(evi_finder_url, evi_finder_loc)
        mtl_state_dict = torch.load(evi_finder_loc, map_location=device)
        for k, _ in mtl_state_dict.items():
            if k.startswith('bert.'):
                mtl_state_dict = fix_name_bare_bert(mtl_state_dict)
                break
        mtl_module.load_state_dict(mtl_state_dict, strict=False)

        cls_module = BertClassifierHotflip(bert_dir=bert_dir,
                                           tokenizer=tokenizer,
                                           mtl_params=mtl_params,
                                           max_length=max_length,
                                           use_half_precision=False,
                                           return_cls_embedding=True,
                                           load_from_local_ckpt=True)
        if not os.path.isfile(cls_loc):
            os.makedirs(cls_loc[::-1].split('/', 1)[1][::-1], exist_ok=True)
            import urllib

            urllib.request.urlretrieve(cls_url, cls_loc)
        cls_state_dict = torch.load(cls_loc, map_location=device)
        for k, _ in cls_state_dict.items():
            if k.startswith('bert.'):
                cls_state_dict = fix_name_bare_bert(cls_state_dict)
                break
        cls_module.load_state_dict(cls_state_dict, strict=False)

    app.jinja_env.filters['zip'] = zip
    # app.run(host='127.0.0.1', port=8080)
    if not os.path.isfile(ugc_data_fname):
        with open(ugc_data_fname, 'w+', newline='') as fout:
            writer = csv.writer(fout)
            writer.writerow('query url evidence label'.split())

    if not os.path.isfile(mgc_data_fname):
        with open(mgc_data_fname, 'w+', newline='') as fout:
            writer = csv.writer(fout)
            writer.writerow('query url evidence label'.split())

    app.run(host='0.0.0.0', port=8080)