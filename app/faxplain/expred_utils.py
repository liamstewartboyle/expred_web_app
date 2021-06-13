from copy import deepcopy
from itertools import chain

import torch


def preprocess(query, docs, tokenizer, top, max_sentence):
    query = query.split()
    tokenized_q, tokenized_q_token_slides = tokenizer.encode_docs([[query]])
    tokenized_q = tokenized_q[0]
    tokenized_q_token_slide = tokenized_q_token_slides[0]
    docs_clean = [[list(chain.from_iterable([s.split() + ['.'] for s in d.lower().split('.')][:max_sentence]))]
                  for d in docs]
    docs = deepcopy(docs_clean)
    tokenized_docs, tokenized_doc_token_slides = tokenizer.encode_docs(docs)
    tokenized_docs = [list(chain.from_iterable(tokenized_docs[i])) for i in range(top)]
    tokenized_doc_token_slides = [list(chain.from_iterable(tokenized_doc_token_slides[i]))
                                  for i in range(top)]
    return tokenized_q, tokenized_q_token_slide, tokenized_docs, tokenized_doc_token_slides, docs_clean


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
    return queries, new_docs


def adapt_exp_pred(exp, doc):
    exp = exp[:len(doc[0])] + [0] * (len(doc[0]) - len(exp))
    return exp


def merge_subtoken_exp_preds(exp_preds, subtoken_mapping):
    ret = []
    for p, ss in zip(exp_preds, subtoken_mapping):
        p = p.tolist()
        ret.append([max(p[s[0]:s[1]] + [0]) for s in ss])
    return ret