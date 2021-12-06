from itertools import chain
from typing import List

import torch


def preprocess_query(query, tokenizer):
    query = query.split()
    tokenized_q, query_slices = tokenizer.encode_docs([[query]])
    tokenized_q = tokenized_q[0]
    query_slice = query_slices[0]
    return tokenized_q, query_slice


def preprocess(query:str, docs:List[str], tokenizer, basic_tokenizer, top, max_sentence, custom_masks:List[List[int]]=None):
    tokenized_q, query_slice = preprocess_query(query, tokenizer)
    docs_split = [[basic_tokenizer.tokenize(d)] for d in docs]
    tokenized_docs, docs_slice = tokenizer.encode_docs([doc[:max_sentence] for doc in docs_split])
    tokenized_docs = [list(chain.from_iterable(tokenized_docs[i])) for i in range(top)]
    docs_slice = [list(chain.from_iterable(docs_slice[i])) for i in range(top)]
    if custom_masks:
        mask_ret =  []
        for slices, mask in zip(docs_slice, custom_masks):
            mask_ret.append([])
            for s, m in zip(slices, mask):
                mask_ret[-1] += [m] * (s[1] - s[0])
    else:
        mask_ret = None
    return tokenized_q, query_slice, tokenized_docs, docs_slice, docs_split, mask_ret


def extract_docs_masks(queries, docs, masks):
    ret = []
    for q, d, m in zip(queries, docs, masks):
        print(len(q), q)
        print(len(d), d)
        print(len(m), m)
        assert len(m) == len(d) + len(q) + 2
        ret.append(m[len(q) + 2:].type(torch.int64))
    return ret


def apply_masks_to_docs(queries:List[torch.Tensor],
                        docs:List[torch.Tensor],
                        masks:List[torch.Tensor],
                        tokenizer,
                        input_max_length,
                        wildcard='.'):
    encoded_wildcard = tokenizer.convert_tokens_to_ids('.')
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