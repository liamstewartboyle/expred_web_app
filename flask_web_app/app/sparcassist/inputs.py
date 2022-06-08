from typing import Any, List, Union

import torch
from expred import ExpredInput
from torch import Tensor


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

    def apply_position_masks(self,
                             masks: Union[List[Tensor], List[List[int]]],
                             are_subtoken_masks: bool,
                             are_input_masks: bool):
        token_doc_masks, subtoken_doc_masks, subtoken_input_masks = self.compute_other_masks(masks,
                                                                                             are_subtoken_masks,
                                                                                             are_input_masks)
        self.subtoken_input_position_masks = self.select_no_overheads(subtoken_input_masks)
        self.subtoken_doc_position_masks = subtoken_doc_masks
        self.token_doc_position_masks = token_doc_masks
        return self

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
                 ann_ids,
                 span_tokenizer,
                 token_doc_custom_pos_mask=None):
        super().__init__(queries,
                         docs,
                         labels,
                         config,
                         ann_ids,
                         span_tokenizer)
        if token_doc_custom_pos_mask is not None:
            self.token_doc_position_masks = [token_doc_custom_pos_mask]

    @classmethod
    def from_ajax_request_factory(cls, request, basic_tokenizer, cf_config, span_tokenizer):
        orig_query = basic_tokenizer.tokenize(request.json['query'])
        orig_doc = [request.json['doc']]
        orig_label = request.json['label']
        ann_id = request.json['ann_id']
        token_doc_custom_pos_mask = cls._extract_token_doc_custom_mask(request)
        cf_input = cls([orig_query],
                       [orig_doc],
                       [orig_label],
                       cf_config,
                       [ann_id],
                       span_tokenizer,
                       token_doc_custom_pos_mask)
        return cf_input
