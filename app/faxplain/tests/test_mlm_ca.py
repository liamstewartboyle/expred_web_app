from unittest import TestCase

import torch
from transformers import BasicTokenizer

from config import CounterfactualConfig
from counterfact_assist import MLMCounterAssist
from expred.expred.expred import Expred
from inputs import CounterfactualInput
from expred.expred.tokenizer import BertTokenizerWithSpans


class TestMLMCounterfactualAssist(TestCase):
    cf_config = CounterfactualConfig()

    span_tokenizer = BertTokenizerWithSpans.from_pretrained(cf_config.bert_dir)
    basic_tokenizer = BasicTokenizer()

    expred = Expred(cf_config)

    # dataset = Dataset(cf_config.dataset_name, cf_config.dataset_base_dir)

    queries = [['what is the sentiment of this review ?']]
    docs = [['joyful test .'.split()]]
    labels = ['NEG']
    original_input = CounterfactualInput(queries=queries,
                                         docs=docs,
                                         labels=labels,
                                         config=cf_config,
                                         ann_ids=None,
                                         span_tokenizer=span_tokenizer)

    def test_get_alternative_preds(self):
        preds = torch.tensor([[0, 1]])
        tiled_preds = torch.tensor([[0, 0, 1, 1]])
        mlm_cf_assist = MLMCounterAssist(self.cf_config, self.expred)
        ret = mlm_cf_assist.get_alternative_preds(preds, tiled_preds)
        self.assertTrue(torch.equal(ret[0], torch.tensor([[1, 0]])))
        self.assertTrue(torch.equal(ret[1], torch.tensor([[1, 1, 0, 0]])))

    def test_convert_unmasker_res_to_expred_input(self):
        mock_unmasker_results = [
            {'sequence': 'joyful contest .', 'pos': 1, 'token_str': 'contest'},
            {'sequence': 'good test .', 'pos': 0, 'token_str': 'good'}
        ]
        token_doc_rationale_masks = [torch.tensor([1, 0, 0], dtype=torch.long)]
        token_doc_position_masks = [torch.tensor([1, 1, 0], dtype=torch.long)]
        mlm_cf_assist = MLMCounterAssist(self.cf_config, self.expred)

        self.original_input.apply_token_doc_rationale_masks(token_doc_rationale_masks)
        self.original_input.apply_token_doc_position_masks(token_doc_position_masks)
        res = mlm_cf_assist.convert_unmasker_res_to_counterfactual_input(mock_unmasker_results,
                                                                         self.original_input,
                                                                         self.span_tokenizer)
        first_encoded_input = res.expred_inputs[0]
        first_encoded_input_should = torch.tensor(self.span_tokenizer.encode('what is the sentiment of this review ?',
                                                                             'joyful contest .'),
                                                  dtype=torch.long)
        self.assertTrue(torch.equal(first_encoded_input, first_encoded_input_should))

        second_encoded_input = res.expred_inputs[1]
        second_encoded_input_should = torch.tensor(self.span_tokenizer.encode('what is the sentiment of this review ?',
                                                                              'good test .') + [0],
                                                   dtype=torch.long)
        self.assertTrue(torch.equal(second_encoded_input, second_encoded_input_should))

    def test_get_mlm_counterfactual(self):
        alternaive_cls_preds = torch.tensor([0], dtype=torch.long)
        query_prompt = 'the sentiment of this review is negative.'
        input_doc = 'i would say that i feel this is essentially a joyful movie .'.split()
        # input_doc = '. . . . . . this is essentially a joyful movie .'.split()
        token_doc_rationale_mask = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
        token_doc_position_mask = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
        cf_input = CounterfactualInput(queries=['what is the sentiment of this review ?'.split()],
                                       docs=[[input_doc]],
                                       labels=['POS'],
                                       config=self.cf_config,
                                       ann_ids=None,
                                       span_tokenizer=self.span_tokenizer)
        cf_input.apply_token_doc_rationale_masks([token_doc_rationale_mask])
        cf_input.apply_token_doc_position_masks([token_doc_position_mask])

        mlm_cf_assist = MLMCounterAssist(self.cf_config, self.expred)
        counterfactual = mlm_cf_assist.get_mlm_counterfactual(alternaive_cls_preds,
                                                              input_doc,
                                                              cf_input,
                                                              self.span_tokenizer)
        self.assertIn(counterfactual['pos'], {9, 10, 11})
        self.assertNotEqual(counterfactual['token_str'].strip(), input_doc[counterfactual['pos']])

    def test_do_counterfactual_generation(self):
        alternaive_cls_preds = torch.tensor([0], dtype=torch.long)
        query_prompt = 'the sentiment of this review is negative.'
        input_doc = 'i would say that i feel this is essentially a joyful movie .'.split()
        # input_doc = '. . . . . . this is essentially a joyful movie .'.split()
        cf_input = CounterfactualInput(queries=['what is the sentiment of this review ?'.split()],
                                       docs=[[input_doc]],
                                       labels=['POS'],
                                       config=self.cf_config,
                                       ann_ids=None,
                                       span_tokenizer=self.span_tokenizer)

        token_doc_rationale_mask = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
        token_doc_position_mask = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]

        cf_input.apply_token_doc_rationale_masks([token_doc_rationale_mask])
        cf_input.apply_token_doc_position_masks([token_doc_position_mask])
        mlm_cf_assist = MLMCounterAssist(self.cf_config, self.expred)

        ret = mlm_cf_assist._do_counterfactual_generation(cf_input, self.span_tokenizer)
        # pprint(ret)
