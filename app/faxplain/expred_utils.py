from itertools import chain
from typing import List

import torch

from expred.expred.models.mlp_mtl import BertClassifierHotflip, BertMTL
from config import class_names
from preprocess import expred_inputs_preprocess


def preprocess_query(query, tokenizer):
    query = query.split()
    tokenized_q, query_slices = tokenizer.encode_docs([[query]])
    tokenized_q = tokenized_q[0]
    query_slice = query_slices[0]
    return tokenized_q, query_slice


def preprocess(query:str, docs:List[str], tokenizer, basic_tokenizer, top, max_sentence):#, custom_masks:List[List[int]]=None):
    encoded_query, query_slice = preprocess_query(query, tokenizer)
    tokenized_docs = [[basic_tokenizer.tokenize(d)] for d in docs]
    encoded_docs, docs_slice = tokenizer.encode_docs([doc[:max_sentence] for doc in tokenized_docs])
    encoded_docs = [list(chain.from_iterable(encoded_docs[i])) for i in range(top)]
    docs_slice = [list(chain.from_iterable(docs_slice[i])) for i in range(top)]
    # if custom_masks:
    #     mask_ret =  []
    #     for slices, mask in zip(docs_slice, custom_masks):
    #         mask_ret.append([])
    #         for s, m in zip(slices, mask):
    #             mask_ret[-1] += [m] * (s[1] - s[0])
    # else:
    #     mask_ret = None
    encoded_query = torch.tensor(encoded_query, dtype=torch.long)
    encoded_docs = [torch.tensor(d, dtype=torch.long) for d in encoded_docs]
    return encoded_query, query_slice, encoded_docs, docs_slice, tokenized_docs


def extract_docs_masks(queries, docs, masks):
    ret = []
    for q, d, m in zip(queries, docs, masks):
        # print(len(q), q)
        # print(len(d), d)
        # print(len(m), m)
        assert len(m) == len(d) + len(q) + 2
        ret.append(m[len(q) + 2:].type(torch.int64))
    return ret


def apply_masks_to_docs(queries:List[torch.Tensor],
                        docs:List[torch.Tensor],
                        masks:List[torch.Tensor],
                        tokenizer,
                        input_max_length=512,
                        wildcard='.'):
    encoded_wildcard = tokenizer.convert_tokens_to_ids(wildcard)
    doc_max_len = input_max_length - 2 - len(queries[0])
    docs = [d[:doc_max_len] if len(d) >= doc_max_len
            else torch.cat([d, torch.zeros(doc_max_len - len(d)).type(torch.int64)])
            for d in docs]
    if len(masks[0]) == len(docs[0]) or not queries: # masks are for encoded docs only
        doc_masks = masks
    else: # masks are for [CLS] encoded query [SEP] encoded doc
        doc_masks = extract_docs_masks(queries, docs, masks)
    masked_docs = [d * m + encoded_wildcard * (1 - m) for d, m in zip(docs, doc_masks)]
    return masked_docs


def merge_subtoken_exp_preds(exp_preds, subtoken_mapping):
    ret = []
    for p, ss in zip(exp_preds, subtoken_mapping):
        p = p.tolist()
        ret.append([max(p[s[0]:s[1]] + [0]) for s in ss])
    return ret


def mtl_predict(exp_module:BertMTL, queries, docs, tokenizer):
    if not isinstance(queries, list):
        queries = [queries]

    with torch.no_grad():
        exp_module.eval()
        max_length = min(max(map(len, docs)) + 2 + len(queries[0]), 512)
        exp_inputs, attention_masks = expred_inputs_preprocess(queries, docs, tokenizer, max_length=max_length)
        aux_preds, exp_preds = exp_module(inputs=exp_inputs, attention_masks=attention_masks)
    aux_preds = [class_names[torch.argmax(p)] for p in aux_preds]

    hard_exp_preds = torch.round(exp_preds)
    # masked_inputs = apply_masks_to_docs(queries, docs, hard_exp_preds, tokenizer, max_length)
    masked_docs = apply_masks_to_docs(queries, docs, hard_exp_preds, tokenizer, max_length)

    docs_start = len(queries[0]) + 2
    docs_end = [docs_start + len(d) for d in masked_docs]
    hard_exp_preds = [p[docs_start: doc_end] for p, doc_end in zip(hard_exp_preds, docs_end)]
    return aux_preds, hard_exp_preds, masked_docs


def cls_predict(cls_module:BertClassifierHotflip, queries, masked_docs, tokenizer):
    with torch.no_grad():
        cls_module.eval()
        cls_inputs, attention_masks = expred_inputs_preprocess(queries, masked_docs, tokenizer)
        cls_preds, _ = cls_module(inputs=cls_inputs, attention_masks=attention_masks)

    cls_preds = [class_names[torch.argmax(p)] for p in cls_preds]
    return cls_preds


def expred_predict(mtl_module, cls_module, queries, docs, docs_slice, tokenizer):
    aux_preds, hard_exp_preds, masked_docs = mtl_predict(mtl_module, queries, docs, tokenizer)
    cls_preds = cls_predict(cls_module, queries, masked_docs, tokenizer)
    return aux_preds, cls_preds, hard_exp_preds#, docs_clean


def fit_mask_to_decoded_docs(mask, tokens, tokenizer):
    ret_mask = []
    p = 0
    for t in tokens:
        tokenized_t = tokenizer.tokenize(t)
        ret_mask.append(max(mask[p: p+len(tokenized_t)]))
        p += len(tokenized_t)
    return ret_mask