import pickle

from flask import Flask, redirect, render_template, request, url_for
from transformers import BasicTokenizer

from config import CounterfactualConfig
from counterfact_assist import ExpredCounterAssist, HotflipCounterAssist, MLMCounterAssist
from counterfact_result import CounterfactResults
from counterfact_writer import CounterfactWriter
from dataset import Dataset
from debug import get_bogus_pred
from expred.expred.utils import pad_mask_to_doclen
from expred_utils import Expred
from faxplain_utils import dump_quel, machine_rationale_mask_to_html
from inputs import CounterfactualInput, ExpredInput
from preprocess import clean
from search import google_rest_api_search as wiki_search
from tokenizer import BertTokenizerWithSpans
from wiki import get_wiki_docs
import logging

extra = {'app_name': 'counterfactual'}
logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s %(app_name)s : %(message)s')
syslog = logging.StreamHandler()
syslog.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.addHandler(syslog)
logger = logging.LoggerAdapter(logger, extra)

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
    cf_input = ExpredInput([query], [doc], [label], cf_config, ann_id, span_tokenizer)

    _, cls_preds, exp_preds = expred(cf_input)
    cls_pred_name = expred.convert_preds_to_names(cls_preds, cf_input.class_names)[0]
    cf_input.apply_subtoken_input_rationale_masks(exp_preds)

    sentence_wise_exps = cf_input.get_sentence_wise_exps(cf_input.token_doc_rationale_masks)

    return render_template('counterfactual.html',
                           ann_id=ann_id,
                           query=' '.join(cf_input.orig_queries[0]),
                           orig_doc=cf_input.orig_sentences[0],
                           explains=sentence_wise_exps[0],
                           label=cf_input.orig_labels[0],
                           pred=cls_pred_name)


@app.route('/show_example', methods=['GET', 'POST'])
def show_example():
    logger.info(str(request.json))
    writer = CounterfactWriter(request)
    session_id = writer.session_id

    cf_config.update_config_from_ajax_request(request)

    cf_input = CounterfactualInput.from_ajax_request_factory(request, basic_tokenizer, cf_config, span_tokenizer)
    if cf_input.subtoken_input_position_masks is None:
        _, cls_preds, exp_preds = expred(cf_input)
        # print('exp_preds: ', exp_preds)
        cf_input.apply_subtoken_input_rationale_masks(exp_preds)
        cf_input.apply_token_doc_position_masks(cf_input.token_doc_rationale_masks)

    selection_strategy = request.json['selection_strategy']
    if selection_strategy == 'mlm':
        counter_assist = MLMCounterAssist(cf_config, expred)
    else:
        counter_assist = HotflipCounterAssist(cf_config, expred)

    cf_results = counter_assist.geneate_counterfactuals(session_id, cf_input, span_tokenizer)

    writer.write_cf_example(cf_results, cf_config)

    return cf_results.todict()


@app.route('/reg_eval', methods=['POST'])
def register_evaluation():
    writer = CounterfactWriter(request)
    cf_results = CounterfactResults.from_dict(request.json)
    writer.write_evaluation(cf_results, cf_config, request.json['eval'])
    return {'placeholder': None}


if False:
    print('debug, will not load models')
else:
    print("Loading models")

    # TODO: initialize config for each session individually
    cf_config = CounterfactualConfig()

    span_tokenizer = BertTokenizerWithSpans.from_pretrained(cf_config.bert_dir)
    basic_tokenizer = BasicTokenizer()

    expred = Expred(cf_config)

    dataset = Dataset(cf_config.dataset_name, cf_config.dataset_base_dir)

    # counter_assist = HotflipCounterAssist(cf_config, expred)
    counter_assist = MLMCounterAssist(cf_config, expred)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
