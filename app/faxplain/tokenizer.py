from itertools import chain
from typing import List, Dict, Tuple, Union

import os
from transformers import BertTokenizer
from transformers.models.bert.tokenization_bert import BasicTokenizer

# from expred.models.pipeline.bert_pipeline import bert_intern_doc
from eraser_utils import Evidence, Annotation


class BertTokenizerWithSpans(BertTokenizer):
    def __init__(self, *args, **kwargs):
        super(BertTokenizerWithSpans, self).__init__(*args, **kwargs)

    def get_sub_token_spans(self, tokens:List[str]) -> List[List[str]]:
        """ Tokenizes a document and returns [start, end) spans to map the wordpieces back to their source words"""
        spans = []
        span_start = 0
        for token in tokens:
            subtokens = super().tokenize(token)
            span_end = span_start + len(subtokens)
            spans.append((span_start, span_end))
            span_start = span_end
        return spans

    def encode(self, tokens:List[str]) -> Dict[str, Union[Tuple[int, int], List[int]]]:
        encoded_input = []
        input_span = []
        span_start = 0
        for token in tokens:
            subtokens = super().encode(token, add_special_tokens=False)
            encoded_input.extend(subtokens)
            span_end = span_start + len(subtokens)
            input_span.append((span_start, span_end))
            span_start = span_end
        return encoded_input, input_span
    
    def encode_docs_with_spans(self, token_docs:List[List[str]])->Tuple[List[List[int]], List[List[Tuple[int, int]]]]:
        encoded_docs = []
        sub_token_spans = []
        for token_doc in token_docs:
            encoded_doc, spans = self.encode(token_doc)
            encoded_docs.append(encoded_doc)
            sub_token_spans.append(spans)
        return encoded_docs, sub_token_spans

    def encode_annotations(self, annotations):
        ret = []
        for ann in annotations:
            ev_groups = []
            for ev_group in ann.evidences:
                evs = []
                for ev in ev_group:
                    text = list(chain.from_iterable(self.tokenize(w)
                                                    for w in ev.text.split()))
                    if len(text) == 0:
                        continue
                    text = self.encode(text, add_special_tokens=False)
                    evs.append(Evidence(text=tuple(text),
                                        docid=ev.docid,
                                        start_token=ev.start_token,
                                        end_token=ev.end_token,
                                        start_sentence=ev.start_sentence,
                                        end_sentence=ev.end_sentence))
                ev_groups.append(tuple(evs))
            query = list(chain.from_iterable(self.tokenize(w)
                                             for w in ann.query.split()))
            if len(query) > 0:
                query = self.encode(query, add_special_tokens=False)
            else:
                query = []
            ret.append(Annotation(annotation_id=ann.annotation_id,
                                  query=tuple(query),
                                  evidences=frozenset(ev_groups),
                                  classification=ann.classification,
                                  query_type=ann.query_type))
        return ret