from copy import deepcopy
from itertools import chain

import torch
from transformers import BasicTokenizer


def preprocess_query(query, tokenizer):
    query = query.split()
    tokenized_q, query_slices = tokenizer.encode_docs([[query]])
    tokenized_q = tokenized_q[0]
    query_slice = query_slices[0]
    return tokenized_q, query_slice


def preprocess(query, docs, tokenizer, basic_tokenizer, top, max_sentence):
    tokenized_q, query_slice = preprocess_query(query, tokenizer)
    docs_split = [[basic_tokenizer.tokenize(d)]
                  for d in docs]
    tokenized_docs, docs_slice = tokenizer.encode_docs([doc[:max_sentence] for doc in docs_split])
    tokenized_docs = [list(chain.from_iterable(tokenized_docs[i])) for i in range(top)]
    docs_slice = [list(chain.from_iterable(docs_slice[i]))
                                  for i in range(top)]
    return tokenized_q, query_slice, tokenized_docs, docs_slice, docs_split


def mark_evidence(queries, docs, hard_preds, tokenizer, max_length, wildcard='.'):
    wildcard_tensor = tokenizer.convert_tokens_to_ids('.') * torch.ones(max_length).type(torch.int)
    doc_max_len = max_length - 2 - len(queries[0])
    docs = [d[:doc_max_len] if len(d) >= doc_max_len
            else torch.cat([d, torch.zeros(doc_max_len - len(d)).type(torch.int64)])
            for d in docs]
    new_docs = []
    for q, d, e in zip(queries, docs, hard_preds):
        temp = torch.cat([torch.zeros(1).type(torch.int64), q, torch.zeros(1).type(torch.int64), d])
        temp = e * temp + (1 - e) * wildcard_tensor
        new_docs.append(temp[(len(queries[0]) + 2):].type(torch.int64))
    return new_docs


def pad_exp_pred(exp, doc):
    doc_len = len(list(chain.from_iterable(doc)))
    exp = exp[:doc_len] + [0] * (doc_len - len(exp))
    return exp


def merge_subtoken_exp_preds(exp_preds, subtoken_mapping):
    ret = []
    for p, ss in zip(exp_preds, subtoken_mapping):
        p = p.tolist()
        ret.append([max(p[s[0]:s[1]] + [0]) for s in ss])
    return ret