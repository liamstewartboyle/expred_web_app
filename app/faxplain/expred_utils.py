from typing import Tuple, List

import torch
from torch import Tensor, nn

from config import CounterfactualConfig
from expred.expred.models.mlp_mtl import CLSModel, MTLModel
from inputs import ExpredInput


def fit_mask_to_decoded_docs(mask, tokens, tokenizer):
    ret_mask = []
    p = 0
    for t in tokens:
        tokenized_t = tokenizer.tokenize(t)
        ret_mask.append(max(mask[p: p + len(tokenized_t)]))
        p += len(tokenized_t)
    return ret_mask


class Expred(nn.Module):
    def __init__(self, cf_config: CounterfactualConfig) -> None:
        super(Expred, self).__init__()
        self.mtl_module = MTLModel.from_pretrained(cf_config.mtl_config)
        self.cls_module = CLSModel.from_pretrained(cf_config.cls_config)

    @staticmethod
    def convert_preds_to_names(preds: Tensor, class_names: List[str]):
        pred_ids = torch.argmax(preds, dim=-1)
        cls_pred_name = [class_names[pid] for pid in pred_ids]
        return cls_pred_name

    def forward(self, inputs: ExpredInput) -> Tuple[Tensor, Tensor, Tensor]:
        mtl_preds = self.mtl_module(inputs.expred_inputs, inputs.attention_masks)
        aux_preds = mtl_preds['aux_preds']
        hard_exp_preds = torch.round(mtl_preds['exp_preds']).type(torch.int)
        rationale_masks = inputs.select_all_overheads(hard_exp_preds)
        inputs.apply_subtoken_input_rationale_masks(rationale_masks)
        cls_preds = self.cls_module(inputs.masked_inputs, inputs.attention_masks)
        cls_preds = cls_preds['cls_pred']
        return aux_preds, cls_preds, hard_exp_preds
