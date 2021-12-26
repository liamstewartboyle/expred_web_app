import pickle
import csv
from typing import List

from transformers import BasicTokenizer
from flask import Flask, request, redirect, url_for, render_template

from preprocess import clean
from wiki import get_wiki_docs
from debug import get_bogus_pred
from tokenizer import BertTokenizerWithMapping
from search import google_rest_api_search as wiki_search
from expred_utils import apply_masks_to_docs, merge_subtoken_exp_preds, preprocess, mtl_predict, cls_predict, \
    expred_predict, fit_mask_to_decoded_docs
from faxplain_utils import dump_quel, highlight_exp_pred, machine_rationale_mask_to_html, parse_sentence, \
    random_select_data
from models.params import MTLParams
from expred.expred.models.utils import fix_name_bare_bert
from expred.expred.utils import pad_mask_to_doclen
from expred.expred.models.mlp_mtl import BertMTL, BertClassifierHotflip
from expred.expred.apply_hotflip import hotflip
from config import *


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        query = request.form['query']
        query = clean(query)
        global default_ndocs
        default_ndocs = int(request.form['cod'])
        return redirect(url_for('prediction', query=query))
    return render_template('index.html')


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
        ) = preprocess(query, orig_docs, tokenizer, basic_tokenizer, default_ndocs, max_sentence)
        queries = [torch.tensor(tokenized_query[0], dtype=torch.long) for i in range(default_ndocs)]
        docs = [torch.tensor(s, dtype=torch.long) for s in tokenized_docs]
        aux_preds, cls_preds, exp_preds, masked_docs = expred_predict(mtl_module, cls_module, queries, docs, docs_slice, tokenizer)
        exp_preds = merge_subtoken_exp_preds(exp_preds, docs_slice)
        exp_preds = [pad_mask_to_doclen(exp, doc) for exp, doc in zip(exp_preds, docs_split)]
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


def get_mtl_mask(orig_query, orig_doc):
    (
        encoded_queries,
        query_slice,
        encoded_docs,
        docs_slice,
        docs_split,
    ) = preprocess(orig_query, [orig_doc], tokenizer, basic_tokenizer, 1, max_sentence, custom_masks)
    # print(len(encoded_queries[0]))
    # print(len(encoded_docs[0]))
    # print(len(custom_masks[0]) if custom_masks else "None")
    encoded_queries = [torch.tensor(encoded_queries[0], dtype=torch.long)]
    encoded_query = encoded_queries[0]
    encoded_docs = [torch.tensor(encoded_docs[0], dtype=torch.long)]
    encoded_doc = encoded_docs[0]


    # print(max_length)

    # the doc_masks are for docs only (no queries or overhead), thus len(doc_masks[0]) == len(tokenized_docs[0].shape)
    # if custom_masks:
    #     doc_masks = [torch.Tensor(m) for m in custom_masks]
    #     masked_docs = apply_masks_to_docs(encoded_queries, encoded_docs, doc_masks, tokenizer, max_length)
    # else:
    #     _, doc_masks, masked_docs = mtl_predict(mtl_module, encoded_queries, encoded_docs, tokenizer)
    _, masks, masked_docs = mtl_predict(mtl_module, encoded_queries, encoded_docs, tokenizer)
    return masks, masked_docs


def get_hotflip(orig_query:str, orig_doc:str, cls_label, top_poses, custom_masks:List[List[int]]=None):

    doc_masks, masked_docs = get_mtl_mask(orig_query, orig_doc)
    top_poses = min(top_poses, len(masked_docs[0]))
    #TODO: fix these magic numbers
    masked_doc = masked_docs[0]
    doc_mask = doc_masks[0]



    cls_inputs = torch.cat(
        (torch.Tensor([tokenizer.cls_token_id]),
         encoded_query,
         torch.Tensor([tokenizer.sep_token_id]),
         masked_doc)).type(torch.long).unsqueeze(0)
    # mtl_masked_data = ['[CLS]'] + basic_tokenizer.tokenize(orig_query) + ['[SEP]'] + basic_tokenizer.tokenize(orig_doc)
    # hard_ann_masks = torch.cat((torch.zeros(len(encoded_query) + 2),
    #                             torch.ones(len(encoded_doc))), dim=-1)
    # print('cls_input: ', cls_input)
    # print('mtl_masked_sentences: ', mtl_masked_doc)
    # print('exp_preds: ', exp_preds)
    cls_attention_masks = torch.ones_like(cls_inputs).type(torch.float).unsqueeze(0)

    hotflip_res = hotflip(cls_module, cls_inputs, hotflip_masks, cls_attention_masks,
                          cls_label.unsqueeze(0),
                          label_classes=class_names,
                          top_poses=top_poses,
                          max_n_words_to_replace=n_word_replace,
                          device='cpu')
    hotflip_doc = hotflip_res[0]['input'][(len(encoded_query) + 2):]
    hotflip_doc = encoded_doc * (1 - doc_mask) + hotflip_doc * doc_mask
    decoded_doc = tokenizer.decode(hotflip_doc)
    hotflip_res[0]['doc'] = basic_tokenizer.tokenize(decoded_doc)
    hotflip_res[0]['mask'] = fit_mask_to_decoded_docs(doc_masks[0], hotflip_res[0]['doc'], tokenizer)
    for i in range(1, len(hotflip_res)):
        hotflip_doc = hotflip_res[i]['input'][(len(encoded_query) + 2):]
        hotflip_doc = encoded_doc * (1 - doc_mask) + hotflip_doc * doc_mask
        hotflip_res[i]['doc'] = tokenizer.decode(hotflip_doc)

    # print(hotflip_res)
    return hotflip_res


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
        encoded_queries,
        query_slice,
        encoded_docs,
        docs_slice,
        _,
    ) = preprocess(orig_query, [orig_doc], tokenizer, basic_tokenizer, 1, 30)
    encoded_queries = [torch.tensor(encoded_queries[0], dtype=torch.long) for _ in range(default_ndocs)]
    encoded_docs = [torch.tensor(s, dtype=torch.long) for s in encoded_docs]
    aux_preds, cls_preds, exp_preds = expred_predict(mtl_module, cls_module,
                                                     encoded_queries, encoded_docs,
                                                     docs_slice, tokenizer)
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


@app.route('/show_example', methods=['GET', 'POST'])
def show_example():
    if request.method == 'GET':
        orig_query = request.args.get('query').strip()
        orig_doc = request.args.get('doc').strip()
        orig_label = request.args.get('label')
    else:
        orig_query = request.json['query'].strip()
        orig_doc = request.json['doc'].strip()
        orig_label = request.json['label']
    # print(orig_doc)
    cls_label = torch.LongTensor(class_names.index(orig_label))
    hotflip_res = get_hotflip(orig_query, orig_doc, cls_label, top_pos)
    return render_template('show_example.html', hotflip_res=hotflip_res)


@app.route('/doc_history', methods=['POST'])
def doc_history():
    # phase 0: parse
    orig_query = request.json['query'].strip()
    raw_orig_doc_html = request.json['masked_raw_html'].strip()
    orig_label = request.json['label']

    mask_method = request.json['mask_method']
    attr_method = request.json['attr_method']
    gen_method = request.json['gen_method']

    gramma_res = request.json['gramma_res']
    allow_ins = request.json['ins']
    allow_del = request.json['del']

    cls_label = torch.LongTensor(class_names.index(orig_label))
    ret = parse_sentence(raw_orig_doc_html, ret_custom_mask=True)
    orig_doc = ret['doc']
    masks = [ret['mask']]

    # phase 0.5: preprocess
    (
        encoded_query,
        query_slice,
        encoded_docs,
        docs_slice,
        _
    ) = preprocess(orig_query, [orig_doc], tokenizer, basic_tokenizer, 1, max_sentence)

    # phase 1: masking
    if mask_method == 'expred':
        _, mask, masked_docs = mtl_predict(mtl_module, encoded_query, encoded_docs, tokenizer)
        masks = [mask]
    else:
        max_length = min(max(map(len, encoded_docs)) + 2 + len(encoded_query), 512)
        masked_docs = apply_masks_to_docs(encoded_query, encoded_docs, masks, tokenizer, max_length)
    masks = [torch.cat([
        torch.zeros(len(encoded_query) + 2).type(torch.long),
        pad_mask_to_doclen(mask, doc)
    ]) for doc, mask in zip(encoded_docs, masks)]
    masks = torch.stack(masks, dim=0)
    cls_inputs = [torch.cat(
        (torch.Tensor([tokenizer.cls_token_id]),
         encoded_query,
         torch.Tensor([tokenizer.sep_token_id]),
         d)).type(torch.long).unsqueeze(0) for d in masked_docs]
    cls_attention_masks = torch.ones_like(cls_inputs).type(torch.float).unsqueeze(0)

    hotflip_res = get_hotflip(orig_query, orig_doc, cls_label, top_pos, masks)
    return render_template('doc_history.html', hotflip_res=hotflip_res)


@app.route('/doc-history', methods=['GET'])
def query_history():
    orig_query = request.args.get('query')
    orig_doc = request.args.get('doc')
    (
        encoded_queries,
        query_slice,
        encoded_docs,
        docs_slice,
        _
    ) = preprocess(orig_query, [orig_doc], tokenizer, basic_tokenizer, 1, max_sentence)
    # [0] for there is only one query and only 1 doc for the counterfactual generation
    queries = [torch.tensor(encoded_queries[0], dtype=torch.long)]
    docs = [torch.tensor(encoded_docs[0], dtype=torch.long)]
    cls_preds = cls_predict(cls_module, queries, docs, tokenizer)
    return render_template('doc_history.html', doc_history=[orig_doc], preds=cls_preds)


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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)