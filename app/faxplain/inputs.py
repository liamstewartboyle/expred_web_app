from itertools import chain
from typing import Any, List, Tuple, Union

import torch
from torch import Tensor
from transformers import BertTokenizer

from config import CounterfactualConfig, ExpredConfig
from tokenizer import BertTokenizerWithSpans


class ExpredInput():
    masked_inputs: Tensor

    def class_name_to_id(self, name):
        return self.class_names.index(name)

    @classmethod
    def encode(cls, inputs: Union[List[str], List[List[str]]], tokenizer: BertTokenizerWithSpans):
        encoded_inputs, input_spans = tokenizer.encode_docs_with_spans(inputs)
        encoded_inputs = [torch.LongTensor(i) for i in encoded_inputs]
        return encoded_inputs, input_spans

    def preprocess_labels(self):
        self.cls_labels = [self.class_name_to_id(label) for label in self.orig_labels]
        self.cls_labels = torch.LongTensor(self.cls_labels).reshape((-1, 1))

    def calculate_overhead_masks(self, tokenized_inputs: Tensor, tokenizer: BertTokenizer):
        actual_encoded_docs = []
        overhead_masks = []
        for q, t in zip(self.encoded_queries, tokenized_inputs):
            try:
                subtoken_doc_end = t.tolist().index(tokenizer.pad_token_id) - 1
            except ValueError:
                subtoken_doc_end = len(t) - 1
            actual_encoded_doc = t[len(q) + 2: subtoken_doc_end]
            actual_encoded_docs.append(actual_encoded_doc)
            mask = torch.cat([torch.ones((1, len(q) + 2)),
                              torch.zeros((1, len(actual_encoded_doc))),
                              torch.ones([1, 1])],
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

    def get_encoded_docs(self):
        encoded_docs = []
        for encoded_query, encoded_input, in zip(self.encoded_queries, self.expred_inputs):
            encoded_docs.append(encoded_input[len(encoded_query) + 2: -1])
        return encoded_docs

    def preprocess_set_mask_wildcard_id(self, tokenizer: BertTokenizer):
        self.wildcard_id = tokenizer.convert_tokens_to_ids(self.config.wildcard_token)

    def preprocess(self, tokenizer: BertTokenizerWithSpans):
        self.encoded_queries, self.query_spans = self.encode(self.orig_queries, tokenizer)
        self.encoded_docs, self.subtoken_docs_spans = self.encode(self.orig_docs, tokenizer)
        self.preprocess_labels()
        self.preprocess_concat_queries_docs(tokenizer)
        self.preprocess_set_mask_wildcard_id(tokenizer)

    def split_exp_into_sentences(self, token_doc_exp_pred: Tensor, sentences: List[List[str]]):
        sentence_wise_exp = []
        exp_start = 0
        for sentence_tokens in sentences:
            sentence_wise_exp.append(token_doc_exp_pred[exp_start: exp_start + len(sentence_tokens)])
            exp_start += len(sentence_tokens)
        return sentence_wise_exp

    def get_sentence_wise_exps(self, token_doc_exp_preds: List[List[int]]):
        sentence_wise_exps = []
        for token_doc_exp_pred, sentences in zip(token_doc_exp_preds, self.orig_sentences):
            sentence_wise_exp = self.split_exp_into_sentences(token_doc_exp_pred, sentences)
            sentence_wise_exps.append(sentence_wise_exp)
        return sentence_wise_exps

    def extract_masks_for_subtoken_docs(self, masks: Union[Tensor, List[List[int]]]) -> List[Tensor]:
        docs_masks = []
        for q, d, m in zip(self.encoded_queries, self.encoded_docs, masks):
            mask = m[len(q) + 2:].type(torch.int)
            max_doc_len = self.config.max_input_len - len(q) - 3
            if len(d) > max_doc_len:
                mask = torch.cat([mask[:-1], torch.zeros(len(d) - max_doc_len).type(torch.int)], dim=-1)
            else:
                mask = mask[:max_doc_len]
            docs_masks.append(mask)
        return docs_masks

    def select_all_overheads(self, masks: Tensor) -> Tensor:
        masks = torch.clip(masks + self.overhead_masks, min=0, max=1)
        return masks

    def select_no_overheads(self, masks: Tensor) -> Tensor:
        masks = masks * (1 - self.overhead_masks)
        return masks

    def apply_masks_to_inputs(self, input_masks: Union[Tensor, List[Union[List[int], Tensor]]]) -> List[
        Union[List[int], Tensor]]:
        if isinstance(input_masks, list):
            input_masks = Tensor(input_masks)
        self.input_masks = input_masks
        self.masked_inputs = self.expred_inputs * self.input_masks + self.wildcard_id * (1 - self.input_masks)
        self.masked_inputs = self.masked_inputs.type(torch.long)

    @classmethod
    def _pool_subtoken_explain(cls, subtoken_doc_explain: List[int], subtoken_doc_spans: List[Tuple[int, int]]) -> List[
        int]:
        token_doc_explain = []
        for subtoken_span in subtoken_doc_spans:
            pooled_token_exp = max(subtoken_doc_explain[subtoken_span[0]: subtoken_span[1]])
            pooled_token_exp = pooled_token_exp
            token_doc_explain.append(pooled_token_exp)
        return token_doc_explain

    def pool_subtoken_docs_explations(self, subtoken_doc_explains: List[Tensor]) -> List[List[int]]:
        token_docs_explains = []
        for subtoken_doc_explain, subtoken_doc_spans in zip(subtoken_doc_explains, self.subtoken_docs_spans):
            subtoken_doc_explain = subtoken_doc_explain.tolist()
            token_docs_explains.append(ExpredInput._pool_subtoken_explain(subtoken_doc_explain, subtoken_doc_spans))
        return token_docs_explains

    def concat_overead_subtoken_mask(self, subtoken_doc_mask):
        return [1] * (len(self.encoded_queries[0]) + 2) + subtoken_doc_mask + [1]

    def __init__(self, queries, docs, labels, config: ExpredConfig, ann_id: str,
                 span_tokenizer: BertTokenizerWithSpans):
        self.orig_queries = queries
        self.orig_sentences: List[List[List[str]]] = docs
        self.orig_docs = [list(chain.from_iterable(doc)) for doc in docs]
        self.orig_labels = labels
        self.config = config
        self.class_names = config.class_names
        self.ann_id = ann_id
        self.preprocess(span_tokenizer)


# TODO: separate the cf_results etc. from the "input" context
class CounterfactualInput(ExpredInput):
    input_mask: List[int]
    custom_doc_masks = None
    custom_input_masks = None
    counterfactual_results = None

    def update_custom_masks(self, subtoken_doc_mask, subtoken_input_mask):
        self.custom_doc_masks = torch.Tensor([subtoken_doc_mask])
        self.custom_input_masks = torch.Tensor([subtoken_input_mask])

    def tile_attention_masks(self, top_poses):
        self.tiled_attention_masks = torch.tile(self.attention_masks, [1, top_poses]).reshape((top_poses, -1))

    def _concat_query_doc_masks(self):
        query_and_overhead_masks = torch.ones([1, self.encoded_queries.shape[-1] + 2]).type(torch.int)
        return torch.concat((query_and_overhead_masks, self.custom_doc_masks), dim=-1)

    def apply_special_restriction(self, subtoken_doc_masks: Tensor) -> Tensor:
        # TODO: grammatic restriction
        return subtoken_doc_masks

    @classmethod
    def _extract_custom_doc_mask(cls, request) -> Union[List[int], Any]:
        return request.json['custom_mask']

    @staticmethod
    def expand_to_subtoken_mask(token_mask, subtoken_spans):
        assert len(token_mask) == len(subtoken_spans)
        ret = []
        for mask, (start, end) in zip(token_mask, subtoken_spans):
            ret += [mask] * (end - start)
        return ret

    def update_custom_masks_from_ajax_request(self, request):
        if request.json['use_custom_mask']:
            custom_doc_mask = CounterfactualInput._extract_custom_doc_mask(request)
            custom_subtoken_doc_mask = self.expand_to_subtoken_mask(custom_doc_mask, self.subtoken_docs_spans[0])
            custom_subtoken_input_mask = self.concat_overead_subtoken_mask(custom_subtoken_doc_mask)
            self.update_custom_masks(custom_subtoken_doc_mask, custom_subtoken_input_mask)

    def __init__(self, request, basic_tokenizer, cf_config, span_tokenizer):
        orig_query = basic_tokenizer.tokenize(request.json['query'])
        orig_doc = [request.json['doc']]
        orig_label = request.json['label']
        ann_id = request.json['ann_id']
        super().__init__([orig_query],
                         [orig_doc],
                         [orig_label],
                         cf_config,
                         ann_id,
                         span_tokenizer)

        self.update_custom_masks_from_ajax_request(request)
