from itertools import chain
from typing import Any, List, Tuple, Type, Union

import torch
from torch import Tensor, LongTensor
from transformers import BertTokenizer
from transformers.models.bert.tokenization_bert import BasicTokenizer

from config import CountefactualConfig, ExpredConfig
from tokenizer import BertTokenizerWithSpans


class ExpredInput():
    masked_inputs: Tensor
    def __init__(self, queries, docs, labels, config:ExpredConfig):
        self.orig_queries = queries
        self.orig_sentences:List[List[List[str]]] = docs
        self.orig_docs = [list(chain.from_iterable(doc)) for doc in docs]
        self.orig_labels = labels
        self.config = config
        self.class_name = config.class_name
        
    def class_name_to_id(self, name):
        return self.class_name.index(name)
    
    def encode(self, inputs:Union[List[str], List[List[str]]], tokenizer:BertTokenizerWithSpans):
        encoded_inputs, input_spans = tokenizer.encode_docs_with_spans(inputs)            
        encoded_inputs = [torch.LongTensor(i) for i in encoded_inputs]
        return encoded_inputs, input_spans
    
    def preprocess_labels(self):
        self.cls_labels = [self.class_name_to_id(label) for label in self.orig_labels]
        self.cls_labels = torch.LongTensor(self.cls_labels).reshape((-1, 1))
        
    def concat_queries_docs(self, tokenizer:BertTokenizerWithSpans):
        tokenize_res = tokenizer([' '.join(q) for q in self.orig_queries], [' '.join(d) for d in self.orig_docs],
                                 max_length=self.config.max_input_len,
                                 padding=True, truncation=True, return_tensors='pt')
        self.expred_inputs = tokenize_res['input_ids']
        self.attention_masks = tokenize_res['attention_mask']
        
    def set_mask_wildcard_id(self, tokenizer:BertTokenizer):
        self.wildcard_id = tokenizer.convert_tokens_to_ids(self.config.wildcard_token)
        
    def preprocess(self, tokenizer:BertTokenizerWithSpans):
        self.encoded_queries, self.query_spans = self.encode(self.orig_queries, tokenizer)
        self.encoded_docs, self.subtoken_docs_spans = self.encode(self.orig_docs, tokenizer)
        self.preprocess_labels()
        
        self.concat_queries_docs(tokenizer)
        
        self.set_mask_wildcard_id(tokenizer)
        
    def split_exp_into_sentences(self, token_doc_exp_pred:Tensor, sentences: List[List[str]]):
        sentence_wise_exp = []
        exp_start = 0
        for sentence_tokens in sentences:
            sentence_wise_exp.append(token_doc_exp_pred[exp_start: exp_start + len(sentence_tokens)])
            exp_start += len(sentence_tokens)
        return sentence_wise_exp
    
    def get_sentence_wise_exps(self, token_doc_exp_preds:Tensor):
        sentence_wise_exps = []
        for token_doc_exp_pred, sentences in zip(token_doc_exp_preds, self.orig_sentences):
            sentence_wise_exp = self.split_exp_into_sentences(token_doc_exp_pred, sentences)
            sentence_wise_exps.append(sentence_wise_exp)
        return sentence_wise_exps
        
    def extract_masks_for_subtoken_docs(self, masks):
        docs_masks = []
        for q, d, m in zip(self.encoded_queries, self.encoded_docs, masks):
            mask = m[len(q) + 2:].type(torch.int)
            input_doc_len = self.config.max_input_len - len(q) - 3
            if len(d) > input_doc_len:
                mask = torch.cat([mask[:-1], torch.zeros(len(d) - input_doc_len).type(torch.int)], dim=-1)
            else:
                mask = mask[:input_doc_len]
            docs_masks.append(mask)
        return docs_masks

    def apply_masks_to_inputs(self, input_masks: Union[Tensor, List[Union[List[int], Tensor]]])->List[Union[List[int], Tensor]]:
        """This is a tool method

        Args:
            inputs (List[Union[List[int], Tensor]]): The inputs can be either the concatenation of query and the doc,
                                                     or be the stand-alon docs. in this case the mask should be at first
                                                     trimmed using the function extact_masks_for_docs()
            masks (Union[Tensor, List[Union[List[int], Tensor]]]): 1 or 0, binary masks,
                                                                   1: this token is preserved
                                                                   0: this token is to replace by the wildcard

        Returns:
            List[Union[List[int], Tensor]]: Masked inputs, the token to be masked are replaced by the wildcard id
        """
        if isinstance(input_masks, list):
            self.input_masks = Tensor(input_masks)
        else:
            self.input_masks = input_masks
        self.masked_inputs = self.expred_inputs * self.input_masks + self.wildcard_id * (1 - self.input_masks)
        self.masked_inputs = self.masked_inputs.type(torch.long)

    @classmethod
    def _pool_subtoken_explain(cls, subtoken_doc_explain:List[int], subtoken_doc_spans:List[Tuple[int, int]]):
        token_doc_explain = []
        for subtoken_span in subtoken_doc_spans:
            pooled_token_exp = max(subtoken_doc_explain[subtoken_span[0] : subtoken_span[1]])
            pooled_token_exp = pooled_token_exp
            token_doc_explain.append(pooled_token_exp)
        return token_doc_explain
    
    def pool_subtoken_docs_explains(self, subtoken_doc_explains:Tensor):
        token_docs_explains = []
        for subtoken_doc_explain, subtoken_doc_spans in zip(subtoken_doc_explains, self.subtoken_docs_spans):
            subtoken_doc_explain = subtoken_doc_explain.tolist()
            token_docs_explains.append(ExpredInput._pool_subtoken_explain(subtoken_doc_explain, subtoken_doc_spans))
        return token_docs_explains

    
class CounterfactualInput(ExpredInput):
    input_mask: List[int]
    custom_doc_masks = None
    custom_input_masks = None
    def __init__(self,
                 query:str,
                 doc:List[List[str]],
                 label: str,
                 cf_config: CountefactualConfig,
                 custom_doc_mask: List[int]=None) -> None:
        super().__init__([query], [doc], [label], cf_config)
        if custom_doc_mask:
            self.custom_doc_masks = [custom_doc_mask]
            self.custom_input_masks = self._concat_query_doc_masks()
        
    # def preprocess(self, tokenizer: BertTokenizerWithSpans, cf_conf: CountefactualConfig):
    #     super().preprocess(tokenizer, cf_conf)
    #     self.encoded_doc = self.encoded_docs[0]
    #     self.subtoken_doc_spans = self.subtoken_docs_spans[0]
        
    #     self.encoded_query = self.encoded_queries[0]
    #     self.subtoken_query_spans = self.subtoken_queries_spans[0]
        
    #     self.orig_label = self.orig_labels[0]
    #     self.cls_label = self.cls_labels[0]
        
    #     self.attention_mask = self.attention_masks[0]
         
    def tile_attention_masks(self, top_poses):
        self.tiled_attention_masks = torch.tile(self.attention_masks, [1, top_poses]).reshape((-1, self.max_input_len))
       
    def _concat_query_doc_masks(self):
        query_and_overhead_masks = torch.ones([1, self.encoded_queries.shape[-1] + 2]).type(torch.int)
        return torch.concat((query_and_overhead_masks, self.custom_doc_masks), dim=-1)
        
    @classmethod
    def _parse_custom_mask(cls, custom_mask_str:str) -> Tensor:
        #TODO
        # also notice that the input custom doc mask is token-wize, need to expand it to subtoken-wise
        raise NotImplementedError
    
    @classmethod
    def _extract_custom_doc_mask(cls, request)->Union[List[int], Any]:
        if request.json['use_custom_mask']:
            return cls._parse_custom_mask(request.json['custom_mask'])
        return None
    
    @classmethod
    def from_ajax_request(cls, request, basic_tokenizer, cf_config):
        orig_query = basic_tokenizer.tokenize(request.json['query'])
        orig_doc = [request.json['doc']]
        orig_label = request.json['label']

        custom_docs_mask = cls._extract_custom_doc_mask(request)
        return CounterfactualInput(orig_query, [orig_doc], orig_label, cf_config, custom_docs_mask)
