from typing import Tuple

import torch
from torch import Tensor, nn

from config import CounterfactualConfig
from expred.expred.models.mlp_mtl import CLSModel, MTLModel
from inputs import ExpredInput

# def mtl_predict(exp_module:MTLModel, queries, docs, tokenizer):
#     if not isinstance(queries, list):
#         queries = [queries]

#     with torch.no_grad():
#         exp_module.eval()
#         max_length = min(max(map(len, docs)) + 2 + len(queries[0]), 512)
#         exp_inputs, attention_masks = expred_inputs_preprocess(queries, docs, tokenizer, max_length=max_length)
#         aux_preds, exp_preds = exp_module(inputs=exp_inputs, attention_masks=attention_masks)
#     aux_preds = [class_names[torch.argmax(p)] for p in aux_preds]

#     hard_exp_preds = torch.round(exp_preds)
#     # masked_inputs = apply_masks_to_docs(queries, docs, hard_exp_preds, tokenizer, max_length)
#     masked_docs = apply_masks_to_docs(queries, docs, hard_exp_preds, tokenizer, max_length)

#     docs_start = len(queries[0]) + 2
#     docs_end = [docs_start + len(d) for d in masked_docs]
#     hard_exp_preds = [p[docs_start: doc_end] for p, doc_end in zip(hard_exp_preds, docs_end)]
#     return aux_preds, hard_exp_preds, masked_docs


# def cls_predict(cls_module:CLSModel, queries, masked_docs, tokenizer):
#     with torch.no_grad():
#         cls_module.eval()
#         cls_inputs, attention_masks = expred_inputs_preprocess(queries, masked_docs, tokenizer)
#         cls_pred, _ = cls_module(inputs=cls_inputs, attention_masks=attention_masks)

#     cls_pred = [class_names[torch.argmax(p)] for p in cls_pred]
#     return cls_pred


# def expred_predict(mtl_module, cls_module, queries, docs, docs_slice, tokenizer):
#     aux_preds, hard_exp_preds, masked_docs = mtl_predict(mtl_module, queries, docs, tokenizer)
#     cls_pred = cls_predict(cls_module, queries, masked_docs, tokenizer)
#     return aux_preds, cls_pred, hard_exp_preds#, docs_clean


def fit_mask_to_decoded_docs(mask, tokens, tokenizer):
    ret_mask = []
    p = 0
    for t in tokens:
        tokenized_t = tokenizer.tokenize(t)
        ret_mask.append(max(mask[p: p+len(tokenized_t)]))
        p += len(tokenized_t)
    return ret_mask

    
class Expred(nn.Module):
    def __init__(self, cf_config:CounterfactualConfig) -> None:
        super(Expred, self).__init__()
        self.mtl_module = MTLModel.from_pretrained(cf_config.mtl_config)
        self.cls_module = CLSModel.from_pretrained(cf_config.cls_config)

    def forward(self, inputs:ExpredInput)->Tuple[Tensor, Tensor, Tensor]:
        mtl_preds = self.mtl_module(inputs.expred_inputs, inputs.attention_masks)
        aux_preds = mtl_preds['aux_preds']
        hard_exp_preds = torch.round(mtl_preds['exp_preds']).type(torch.int)
        rationale_masks = inputs.select_all_overheads(hard_exp_preds)
        inputs.apply_masks_to_inputs(rationale_masks)
        cls_preds = self.cls_module(inputs.masked_inputs, inputs.attention_masks)
        cls_preds = cls_preds['cls_pred']
        return aux_preds, cls_preds, hard_exp_preds
    