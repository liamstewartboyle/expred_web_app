from itertools import chain
from typing import Any, List, Tuple, Union

import torch
from torch import Tensor
from transformers import BertTokenizer

from config import ExpredConfig
from tokenizer import BertTokenizerWithSpans


class ExpredInput:
    masked_inputs: Tensor

    encoded_docs = None
    docs_spans = None

    encoded_queries = None
    query_spans = None

    subtoken_input_rationale_masks = None
    subtoken_doc_rationale_masks = None

    token_doc_rationale_masks = None

    def class_name_to_id(self, name):
        return self.class_names.index(name)

    def calculate_overhead_masks(self, tokenized_inputs: Tensor, tokenizer: BertTokenizer):
        actual_encoded_docs = []
        overhead_masks = []
        max_padded_len = tokenized_inputs.shape[-1]
        for q, t in zip(self.encoded_queries, tokenized_inputs):
            try:
                subtoken_doc_end = t.tolist().index(tokenizer.pad_token_id) - 1
            except ValueError:
                subtoken_doc_end = len(t) - 1
            actual_encoded_doc = t[len(q) + 2: subtoken_doc_end]
            actual_encoded_docs.append(actual_encoded_doc)
            padding_len = max_padded_len - len(q) - 3 - len(actual_encoded_doc)
            mask = torch.cat([torch.ones((1, len(q) + 2)),
                              torch.zeros((1, len(actual_encoded_doc))),
                              torch.ones([1, 1]),
                              torch.ones(1, padding_len)],
                             dim=-1)
            overhead_masks.append(mask)
        self.actual_encoded_docs = actual_encoded_docs
        self.overhead_masks = torch.cat(overhead_masks, dim=0).type(torch.int)

    def preprocess_concat_queries_docs(self, tokenizer: BertTokenizerWithSpans) -> None:
        tokenized_res = tokenizer([' '.join(q) for q in self.orig_queries], [' '.join(d) for d in self.orig_docs],
                                  max_length=self.config.max_input_len,
                                  padding=True, truncation=True, return_tensors='pt')
        self.calculate_overhead_masks(tokenized_res['input_ids'], tokenizer)
        self.expred_inputs = tokenized_res['input_ids']
        self.attention_masks = tokenized_res['attention_mask']

    def extract_subtoken_docs(self, subtoken_inputs):
        subtoken_docs = []
        for subtoken_query, subtoken_input in zip(self.encoded_queries, subtoken_inputs):
            subtoken_doc = subtoken_input[subtoken_query.shape[-1] + 2 : -1]
            subtoken_docs.append(subtoken_doc)
        return subtoken_docs

    def preprocess_labels(self):
        cls_labels = [self.class_name_to_id(label) for label in self.orig_labels]
        self.cls_labels = torch.tensor(cls_labels, dtype=torch.long).reshape((-1, 1))

    @staticmethod
    def encode(inputs: Union[List[str], List[List[str]]], tokenizer: BertTokenizerWithSpans):
        encoded_inputs, input_spans = tokenizer.encode_docs_with_spans(inputs)
        encoded_inputs = [torch.tensor(i, dtype=torch.long) for i in encoded_inputs]
        return encoded_inputs, input_spans

    def preprocess(self, tokenizer: BertTokenizerWithSpans):
        self.encoded_queries, self.query_spans = self.encode(self.orig_queries, tokenizer)
        self.encoded_docs, self.docs_spans = self.encode(self.orig_docs, tokenizer)
        self.preprocess_labels()
        self.preprocess_concat_queries_docs(tokenizer)

    @staticmethod
    def split_exp_into_sentences(token_doc_exp_pred: List[int], sentences: List[List[str]]):
        sentence_wise_exp = []
        exp_start = 0
        for sentence_tokens in sentences:
            sentence_wise_exp.append(token_doc_exp_pred[exp_start: exp_start + len(sentence_tokens)])
            exp_start += len(sentence_tokens)
        return sentence_wise_exp

    def get_sentence_wise_exps(self, token_docs_exp_preds: List[List[int]]):
        sentence_wise_exps = []
        for token_doc_exp_pred, sentences in zip(token_docs_exp_preds, self.orig_sentences):
            sentence_wise_exp = self.split_exp_into_sentences(token_doc_exp_pred, sentences)
            sentence_wise_exps.append(sentence_wise_exp)
        return sentence_wise_exps

    def select_all_overheads(self, masks: Tensor) -> List[Tensor]:
        masks = [torch.clip(m + om, min=0, max=1) for m, om in zip(masks, self.overhead_masks)]
        return masks

    def select_no_overheads(self, masks: Tensor) -> List[Tensor]:
        masks = [m * (1 - om) for m, om in zip(masks, self.overhead_masks)]
        return masks

    @staticmethod
    def maybe_padding(Xs: List[Tensor], padding=0) -> Tensor:
        max_len = max(x.shape[-1] for x in Xs)
        ret = []
        for x in Xs:
            if x.shape[-1] < max_len:
                p = torch.cat([x, torch.tensor([padding] * (max_len - x.shape[-1]), dtype=torch.long)])
            else:
                p = x
            ret.append(p)
        return torch.stack(ret, dim=0)

    def mask_subtoken_inputs(self, masks: Tensor):
        w = self.wildcard_id
        _input = self.expred_inputs
        return (_input * masks + w * (1 - masks)).type(torch.long)

    def apply_token_doc_rationale_masks(self, token_doc_rationale_masks: Union[
        Tensor, List[Union[List[int], Tensor]]]):
        self.token_doc_rationale_masks = token_doc_rationale_masks
        # print([len(s) for s in self.docs_spans])
        # print(self.encoded_docs[-3])
        # print(self.encoded_docs[-2])
        self.subtoken_doc_rationale_masks = self.expand_token_masks(token_doc_rationale_masks, self.docs_spans)
        self.subtoken_input_rationale_masks = self.pad_to_input_subtoken_masks(self.subtoken_doc_rationale_masks)
        self.masked_inputs = self.mask_subtoken_inputs(self.subtoken_input_rationale_masks)

    def apply_subtoken_doc_rationale_masks(self, subtoken_doc_rationale_masks: List[Tensor]):
        self.token_doc_rationale_masks = self.pool_subtoken_masks(subtoken_doc_rationale_masks, self.docs_spans)
        self.subtoken_doc_rationale_masks = subtoken_doc_rationale_masks
        self.subtoken_input_rationale_masks = self.pad_to_input_subtoken_masks(subtoken_doc_rationale_masks)
        self.masked_inputs = self.mask_subtoken_inputs(self.subtoken_input_rationale_masks)

    def apply_subtoken_input_rationale_masks(self, subtoken_input_rationale_masks: List[Tensor]):
        subtoken_doc_rationale_masks = self.trim_to_doc_rationale_masks(subtoken_input_rationale_masks)

        self.token_doc_rationale_masks = self.pool_subtoken_masks(subtoken_doc_rationale_masks, self.docs_spans)
        self.subtoken_doc_rationale_masks = subtoken_doc_rationale_masks
        self.subtoken_input_rationale_masks = self.maybe_padding(subtoken_input_rationale_masks)
        self.masked_inputs = self.mask_subtoken_inputs(self.subtoken_input_rationale_masks)

    def pad_to_input_subtoken_masks(self, subtoken_doc_masks: List[Tensor]) -> Tensor:
        len_query = self.encoded_queries[0].shape[0]
        unpadded_input_rationale_masks = [torch.cat([torch.ones(len_query + 2), mask, torch.ones(1)])
                                          for mask in subtoken_doc_masks]
        return self.maybe_padding(unpadded_input_rationale_masks)

    def trim_to_doc_rationale_masks(self, subtoken_input_masks) -> List[Tensor]:
        docs_masks = []
        for q, d, m in zip(self.encoded_queries, self.encoded_docs, subtoken_input_masks):
            mask = m[len(q) + 2:].type(torch.int)
            max_doc_len = self.config.max_input_len - len(q) - 3
            if len(d) > max_doc_len:
                mask = torch.cat([mask[:-1], torch.zeros(len(d) - max_doc_len).type(torch.int)], dim=-1)
            else:
                mask = mask[:max_doc_len]
            docs_masks.append(mask)
        return docs_masks

    @staticmethod
    def expand_token_masks(token_masks, subtoken_spans):
        rets = []
        # print(token_masks, subtoken_spans)
        for mask, spans in zip(token_masks, subtoken_spans):
            # print(len(mask), len(spans))
            # print(mask, spans)
            assert len(mask) == len(spans)
            mask_ret = []
            for m, (start, end) in zip(mask, spans):
                mask_ret += [m] * (end - start)
            rets.append(torch.tensor(mask_ret, dtype=torch.long))
        return rets

    @staticmethod
    def pool_subtoken_masks(subtoken_masks: List[Tensor], subtoken_spans) -> List[List[int]]:

        def _pool_subtoken_mask(subtoken_mask: List[int], spans: List[Tuple[int, int]]) -> List[int]:
            token_doc_explain = []
            for subtoken_span in spans:
                pooled_token_exp = max(subtoken_mask[subtoken_span[0]: subtoken_span[1]])
                pooled_token_exp = pooled_token_exp
                token_doc_explain.append(pooled_token_exp)
            return token_doc_explain

        token_masks = []
        for subtoken_mask, spans in zip(subtoken_masks, subtoken_spans):
            if isinstance(subtoken_mask, Tensor):
                subtoken_mask = subtoken_mask.tolist()
            token_masks.append(_pool_subtoken_mask(subtoken_mask, spans))
        return token_masks

    def __init__(self,
                 queries, docs, labels,
                 config: ExpredConfig, ann_id: Union[str, Any],
                 span_tokenizer: BertTokenizerWithSpans):
        # the actual_encoded_docs in the input is sometimes different
        # from encoded_docs because of the input length restriction
        self.expred_inputs = None
        self.overhead_masks = None
        self.actual_encoded_docs = None
        self.cls_labels = None
        self.orig_queries = queries
        self.orig_sentences: List[List[List[str]]] = docs
        self.orig_docs = [list(chain.from_iterable(doc)) for doc in docs]
        self.orig_labels = labels
        self.ann_id = ann_id

        self.config = config
        self.class_names = config.class_names

        self.wildcard_token = self.config.wildcard_token
        self.wildcard_id = span_tokenizer.convert_tokens_to_ids(self.wildcard_token)

        self.preprocess(span_tokenizer)


# TODO: separate the cf_results etc. from the "input" context
class CounterfactualInput(ExpredInput):
    counterfactual_results = None
    tiled_attention_masks = None
    subtoken_input_position_masks = None
    subtoken_doc_position_masks = None
    token_doc_position_masks = None

    def tile_attention_masks(self, top_poses):
        self.tiled_attention_masks = torch.tile(self.attention_masks, [1, top_poses]).reshape((top_poses, -1))

    def apply_special_restriction(self, subtoken_doc_masks: Tensor) -> Tensor:
        # TODO: grammatic restriction
        return subtoken_doc_masks

    def apply_token_doc_position_masks(self, token_doc_position_masks: List[List[int]]):
        self.token_doc_position_masks = token_doc_position_masks
        self.subtoken_doc_position_masks = self.expand_token_masks(token_doc_position_masks, self.docs_spans)
        temp = self.pad_to_input_subtoken_masks(self.subtoken_doc_position_masks)
        self.subtoken_input_position_masks = self.select_no_overheads(temp)

    def apply_subtoken_input_position_masks(self, subtoken_input_position_masks: List[Tensor]):
        self.subtoken_input_position_masks = subtoken_input_position_masks
        self.subtoken_doc_position_masks = self.trim_to_doc_rationale_masks(subtoken_input_position_masks)
        self.token_doc_position_masks = self.pool_subtoken_masks(self.subtoken_doc_position_masks, self.docs_spans)

    @staticmethod
    def _extract_token_doc_custom_mask(request) -> Union[List[int], Any]:
        if request.json['use_custom_mask']:
            return request.json['custom_mask']
        else:
            return None

    def __init__(self,
                 queries,
                 docs,
                 labels,
                 config,
                 ann_id,
                 span_tokenizer):
        super().__init__(queries,
                         docs,
                         labels,
                         config,
                         ann_id,
                         span_tokenizer)

    @classmethod
    def from_ajax_request_factory(cls, request, basic_tokenizer, cf_config, span_tokenizer):
        orig_query = basic_tokenizer.tokenize(request.json['query'])
        orig_doc = [request.json['doc']]
        orig_label = request.json['label']
        ann_id = request.json['ann_id']
        cf_input = cls([orig_query],
                       [orig_doc],
                       [orig_label],
                       cf_config,
                       ann_id,
                       span_tokenizer)

        token_doc_custom_mask = cf_input._extract_token_doc_custom_mask(request)
        if token_doc_custom_mask is not None:
            cf_input.apply_token_doc_rationale_masks([token_doc_custom_mask])
            subtoken_input_position_masks = cf_input.select_no_overheads(cf_input.subtoken_input_rationale_masks)
            cf_input.apply_subtoken_input_position_masks(subtoken_input_position_masks)
        return cf_input
