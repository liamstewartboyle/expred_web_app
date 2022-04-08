from unittest import TestCase

import torch
from transformers import BasicTokenizer

from config import CounterfactualConfig
from counterfact_assist import MLMCounterAssist
from expred.expred.expred import Expred
from expred.expred.inputs import ExpredInput
from expred.expred.tokenizer import BertTokenizerWithSpans


class TestExpredInput(TestCase):
    cf_config = CounterfactualConfig()

    span_tokenizer = BertTokenizerWithSpans.from_pretrained(cf_config.bert_dir)
    basic_tokenizer = BasicTokenizer()

    expred = Expred(cf_config)

    # dataset = Dataset(cf_config.dataset_name, cf_config.dataset_base_dir)

    mlm_cf_assist = MLMCounterAssist(cf_config, expred)

    queries = [['what is that ?']]
    docs = [[['that is bad .']]]
    labels = ['NEG']
    original_input = ExpredInput(queries=queries,
                                 docs=docs,
                                 labels=labels,
                                 config=cf_config,
                                 ann_ids=None,
                                 span_tokenizer=span_tokenizer)

    alternaive_cls_preds = torch.tensor([0], dtype=torch.long)
    query_prompt = 'the sentiment of this review is negative.'
    input_doc = 'i would say that i feel this is essentially a joyful movie .'.split()
    masked_doc = '. . . . . . this is essentially a joyful movie .'
    encoded_masked_doc = span_tokenizer.encode(masked_doc, add_special_tokens=False)
    token_doc_rationale_mask = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
    subtoken_doc_rationale_mask_should = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
    subtoken_input_rationale_masks_should = torch.tensor(
        [1] * 10 + subtoken_doc_rationale_mask_should + [1], dtype=torch.long)

    token_doc_position_mask = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]

    cf_input = ExpredInput(queries=['what is the sentiment of this review ?'.split()],
                           docs=[[input_doc]],
                           labels=['POS'],
                           config=cf_config,
                           ann_ids=None,
                           span_tokenizer=span_tokenizer)

    encoded_doc_should = [1045, 2052, 2360, 2008, 1045, 2514, 2023, 2003, 7687, 1037, 6569, 3993, 3185, 1012]
    doc_span_should = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 12),
                       (12, 13), (13, 14)]
    encoded_query_should = [2054, 2003, 1996, 15792, 1997, 2023, 3319, 1029]
    query_span_should = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]

    def test_encode(self):
        encoded, spans = ExpredInput.encode([self.input_doc], self.span_tokenizer)
        self.assertSequenceEqual(encoded, [self.encoded_doc_should])
        self.assertSequenceEqual(spans, [self.doc_span_should])

    def test__init__(self):
        self.assertTrue(
            torch.equal(self.cf_input.encoded_queries[0], torch.tensor(self.encoded_query_should, dtype=torch.long)))
        self.assertSequenceEqual(self.cf_input.query_spans, [self.query_span_should])

        self.assertTrue(
            torch.equal(self.cf_input.encoded_docs[0], torch.tensor(self.encoded_doc_should, dtype=torch.long)))
        self.assertSequenceEqual(self.cf_input.docs_spans, [self.doc_span_should])

        self.assertEqual(self.cf_input.cls_labels, torch.tensor([1]))

        expred_inputs_should = torch.tensor(
            [[101] + self.encoded_query_should + [102] + self.encoded_doc_should + [102]])
        self.assertTrue(torch.equal(self.cf_input.expred_inputs, expred_inputs_should))

        attention_masks_should = torch.ones_like(expred_inputs_should).type(torch.long)
        self.assertTrue(torch.equal(self.cf_input.attention_masks, attention_masks_should))

        self.assertTrue(torch.equal(self.cf_input.actual_encoded_docs[0], self.cf_input.encoded_docs[0]))

        overhead_masks_should = torch.tensor([[1] +
                                              [1] * len(self.encoded_query_should) +
                                              [1] +
                                              [0] * len(self.encoded_doc_should) +
                                              [1]], dtype=torch.int)
        self.assertTrue(torch.equal(self.cf_input.overhead_masks, overhead_masks_should))

    def test_expand_doc_rationale_masks(self):
        subtoken_doc_rationale_mask = self.cf_input.expand_token_masks([self.token_doc_rationale_mask],
                                                                       self.cf_input.docs_spans)
        self.assertSequenceEqual(self.subtoken_doc_rationale_mask_should, subtoken_doc_rationale_mask)

    def test_update_and_apply_subtoken_input_rationale_masks(self):
        self.cf_input.apply_subtoken_input_rationale_masks(
            self.subtoken_input_rationale_masks_should.unsqueeze(0))
        self.assertTrue(torch.equal(self.cf_input.subtoken_input_rationale_masks,
                                    self.subtoken_input_rationale_masks_should.unsqueeze(0)))
        masked_inputs_should = torch.tensor(
            [[101] +
             self.encoded_query_should +
             [102] +
             self.encoded_masked_doc +
             [102]], dtype=torch.long)

        self.assertTrue(torch.equal(self.cf_input.masked_inputs, masked_inputs_should))

        self.cf_input.apply_subtoken_input_rationale_masks(
            [self.subtoken_input_rationale_masks_should.tolist()])
        self.assertTrue(
            torch.equal(self.cf_input.subtoken_input_rationale_masks,
                        self.subtoken_input_rationale_masks_should.unsqueeze(0)))
        self.assertTrue(torch.equal(self.cf_input.masked_inputs, masked_inputs_should))

    def test_update_and_apply_token_doc_rationale_masks(self):
        self.cf_input.apply_token_doc_rationale_masks([self.token_doc_rationale_mask])
        self.assertSequenceEqual([self.subtoken_doc_rationale_mask_should], self.cf_input.subtoken_doc_rationale_masks)
        self.assertTrue(torch.equal(self.subtoken_input_rationale_masks_should.unsqueeze(0),
                                    self.cf_input.subtoken_input_rationale_masks))
        self.assertTrue(torch.equal(self.cf_input.subtoken_input_rationale_masks,
                                    self.subtoken_input_rationale_masks_should.unsqueeze(0)))
