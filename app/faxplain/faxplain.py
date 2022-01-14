import pickle

from flask import Flask, redirect, render_template, request, url_for
from transformers import BasicTokenizer

from config import CounterfactualConfig
from counter_assist import ExpredCounterAssist
from counterfact_writer import CounterfactWriter
from dataset import Dataset
from debug import get_bogus_pred
from expred.expred.utils import pad_mask_to_doclen
from expred_utils import Expred
from faxplain_utils import dump_quel, machine_rationale_mask_to_html
from inputs import CounterfactualInput
from preprocess import clean
from search import google_rest_api_search as wiki_search
from tokenizer import BertTokenizerWithSpans
from wiki import get_wiki_docs


class CustomFlask(Flask):
    jinja_options = Flask.jinja_options.copy()
    jinja_options.update(dict(
        # Default is '{{', I'm changing this because Vue.js uses '{{' / '}}'
        variable_start_string='%%',
        variable_end_string='%%',
    ))


app = CustomFlask(__name__)
app.jinja_env.filters['zip'] = zip


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


def get_mtl_mask(encoded_queries, encoded_docs):
    encoded_queries = [torch.tensor(encoded_queries[0], dtype=torch.long)]
    encoded_docs = [torch.tensor(encoded_docs[0], dtype=torch.long)]

    # the doc_masks are for docs only (no queries or overhead), thus len(doc_masks[0]) == len(tokenized_docs[0].shape)
    # if custom_masks:
    #     doc_masks = [torch.Tensor(m) for m in custom_masks]
    #     masked_docs = apply_masks_to_docs(encoded_queries, encoded_docs, doc_masks, tokenizer, max_length)
    # else:
    #     _, doc_masks, masked_docs = mtl_predict(mtl_module, encoded_queries, encoded_docs, tokenizer)
    _, masks, masked_docs = mtl_predict(
        mtl_module, encoded_queries, encoded_docs, span_tokenizer)
    return masks, masked_docs


@app.route('/counterfactual', methods=['GET', 'POST'])
def counterfactual():
    ann_id, query, doc, label = dataset.random_select_data(basic_tokenizer)

    cf_input.init_from_dataset(query, doc, label, cf_config, ann_id)
    cf_input.preprocess(span_tokenizer)

    _, cls_preds, exp_preds = expred(cf_input)
    cls_pred_id = cls_preds[0].tolist().index(max(cls_preds[0]))
    cls_pred = cf_input.class_names[cls_pred_id]

    subtoken_doc_exp_preds = cf_input.extract_masks_for_subtoken_docs(exp_preds)
    token_doc_exp_preds = cf_input.pool_subtoken_docs_explations(subtoken_doc_exp_preds)

    sentence_wise_exps = cf_input.get_sentence_wise_exps(token_doc_exp_preds)

    return render_template('counterfactual.html',
                           query=' '.join(cf_input.orig_queries[0]),
                           orig_doc=cf_input.orig_sentences[0],
                           explains=sentence_wise_exps[0],
                           label=cf_input.orig_labels[0],
                           pred=cls_pred)


@app.route('/show_example', methods=['GET', 'POST'])
def show_example():
    # TODO: flush custom mask for each sentence/each documents
    cf_config.update_config_from_ajax_request(request)
    cf_input.update_from_ajax_request(request, basic_tokenizer, cf_config)
    cf_input.preprocess(span_tokenizer)
    cf_input.update_custom_masks_from_ajax_request(request)

    cf_examples = counter_assist.geneate_counterfactuals(cf_input, span_tokenizer)
    cf_results = {"cf_examples": cf_examples,
                  'ann_id': cf_input.ann_id}
    cf_input.counterfactual_results = cf_results

    writer.write_cf_example(cf_input, cf_config)

    return cf_results


@app.route('/reg_eval', methods=['POST'])
def register_evaluation():
    writer.write_evaluation(cf_input, cf_config, request.json)
    return {'placeholder': None}


if False:
    print('debug, will not load models')
else:
    print("Loading models")

    cf_config = CounterfactualConfig()

    cf_input = CounterfactualInput()

    span_tokenizer = BertTokenizerWithSpans.from_pretrained(cf_config.bert_dir)
    basic_tokenizer = BasicTokenizer()

    expred = Expred(cf_config)

    dataset = Dataset(cf_config.dataset_name, cf_config.dataset_base_dir)

    counter_assist = ExpredCounterAssist(cf_config, expred)

    writer = CounterfactWriter()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
