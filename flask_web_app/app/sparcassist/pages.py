import os

import torch
from expred import BertTokenizerWithSpans, Expred, BertDataset, ExpredInput
from flask import request, Blueprint, render_template
from transformers import BasicTokenizer

from sparcassist import CounterfactWriter, CounterfactualInput, MLMCounterAssist, HotflipCounterAssist, \
    CounterfactResults
from sparcassist.config import CounterfactualConfig
from sparcassist.model import ExpredCounterAssist
from utils.logger import get_logger

sparcassist_pages = Blueprint('sparcassist', __name__, template_folder='templates')

FLASK_ENV = os.getenv('FLASK_ENV')
development = True if (FLASK_ENV is not None and 'development' in FLASK_ENV) else False

logger = get_logger('sparcassist')


@sparcassist_pages.route('/sparcassist', methods=['GET', 'POST'])
def counterfactual():
    ann_id, query, doc, label = dataset.random_select_data(basic_tokenizer)
    cf_input = ExpredInput([query], [doc], [label], cf_config, [ann_id], span_tokenizer).preprocess()

    ret = expred(cf_input)
    cls_preds = ret['cls_preds']['cls_pred']
    cls_pred_name = cf_input.cls_preds_to_class_name(cls_preds)[0]

    sentence_wise_exps = cf_input.get_sentence_wise_exps(cf_input.token_doc_rationale_masks)

    return render_template('sparcassist.html',
                           ann_id=ann_id,
                           query=' '.join(cf_input.orig_queries[0]),
                           orig_doc=cf_input.orig_sentences[0],
                           explains=sentence_wise_exps[0],
                           label=cf_input.orig_labels[0],
                           pred=cls_pred_name)


def apply_cf_input_masks(expred: Expred, cf_input: CounterfactualInput) -> None:
    expred_output = expred(cf_input)
    subtoken_input_rationale_masks = torch.round(expred_output['mtl_preds']['exp_preds'])
    cf_input.apply_rationale_masks(subtoken_input_rationale_masks, are_subtoken_masks=True, are_input_masks=True)

    if cf_input.token_doc_position_masks is None:  # i.e. with custom token-wise doc position masks
        token_doc_position_masks = cf_input.token_doc_rationale_masks
    else:
        token_doc_position_masks = cf_input.token_doc_position_masks
    cf_input.apply_position_masks(token_doc_position_masks, False, False)


def get_assist(model_name: str) -> ExpredCounterAssist:
    if model_name == 'mlm':
        counter_assist = MLMCounterAssist(cf_config, expred)
    else:
        counter_assist = HotflipCounterAssist(cf_config, expred)
    return counter_assist


@sparcassist_pages.route('/show_example', methods=['GET', 'POST'])
def show_example():
    logger.info(str(request.json))
    writer = CounterfactWriter(request)
    session_id = writer.session_id

    cf_config.update_config_from_ajax_request(request)

    cf_input = CounterfactualInput.from_sentence_selection(request,
                                                           basic_tokenizer,
                                                           cf_config,
                                                           span_tokenizer).preprocess()

    apply_cf_input_masks(expred, cf_input)

    counter_assist = get_assist(request.json['selection_strategy'])

    cf_results = counter_assist.generate_counterfactuals(session_id, cf_input, span_tokenizer)

    writer.write_cf_example(cf_results, cf_config)
    return cf_results.to_dict()


@sparcassist_pages.route('/select_alt_word', methods=['GET', 'POST'])
def select_alt_word():
    logger.info(str(request.json))
    writer = CounterfactWriter(request)
    session_id = writer.session_id

    cf_results_first_half = request.json['cf_examples']
    first_prediction = cf_results_first_half['instances'][0]['pred'][0]
    query = request.json['query']
    alt_word_ids = list(map(int, request.json['alt_word_id'].split('.')))
    cf_input = CounterfactualInput.from_cf_example(cf_results_first_half,
                                                   query,
                                                   alt_word_ids,
                                                   basic_tokenizer=basic_tokenizer,
                                                   cf_config=cf_config,
                                                   span_tokenizer=span_tokenizer).preprocess()

    apply_cf_input_masks(expred, cf_input)

    counter_assist = get_assist(request.json['selection_strategy'])

    cf_results_second_half = counter_assist.generate_counterfactuals(session_id, cf_input, span_tokenizer,
                                                                     first_prediction=first_prediction)

    cf_results = CounterfactResults.combine_cf_results(cf_results_first_half,
                                                       alt_word_ids[0],
                                                       cf_results_second_half)

    writer.write_cf_example(cf_results, cf_config)
    return cf_results.to_dict()

@sparcassist_pages.route('/reg_eval', methods=['POST'])
def register_evaluation():
    writer = CounterfactWriter(request)
    cf_results = CounterfactResults.from_dict(request.json)
    writer.write_evaluation(cf_results, cf_config, request.json['eval'])
    return {'placeholder': None}


print("Loading models")
cf_config = CounterfactualConfig(development)
span_tokenizer = BertTokenizerWithSpans.from_pretrained(cf_config.bert_dir)
basic_tokenizer = BasicTokenizer()
expred = Expred(cf_config)
dataset = BertDataset(cf_config.dataset_name)
