from copy import deepcopy
from typing import Dict, List, Union

import torch
from transformers.utils.dummy_tf_objects import TensorFlowBenchmark
from config import CountefactualConfig
from expred.expred.models.mlp_mtl import CLSModel
from expred_utils import Expred
from torch import Tensor, nn

from inputs import CounterfactualInput


def scoring_position_grad(grad_wrt_embd:Tensor, bert_embeddings:Tensor, rationale_masks: Tensor) -> Tensor:
    position_scores = torch.mean(grad_wrt_embd * bert_embeddings, dim=-1) + (1 - rationale_masks) * 1e16  # B*L*1
    return position_scores


def scoring_words_grad(grad_wrt_embd:Tensor, word_embedding:Tensor) -> Tensor:
    word_embedding_t = word_embedding.transpose(1, 2)
    word_scores = grad_wrt_embd.matmul(word_embedding_t)
    return word_scores

position_scoring_methods = {'gradient': scoring_position_grad}
word_scoring_methods = {'gradient': scoring_words_grad}
    
class ExpredCounterAssist():
    def __init__(self, cf_config: CountefactualConfig, model: Expred):
        # self.cf_config = cf_config
        self.number_top_positions = cf_config.number_top_positions
        self.max_count_word_replacement = cf_config.max_count_word_replacement
        self.device = cf_config.device
        self.max_input_len = cf_config.max_input_len
        self.position_scoring_method = cf_config.position_scoring_method
        self.word_scoring_method = cf_config.word_scoring_method
        self.model = model
        self.init_model()
        self.word_embedding = self._get_word_embedding()
        self.loss_criterion = nn.CrossEntropyLoss(reduction="none")
        self.scoring_positions = position_scoring_methods[cf_config.position_scoring_method]
        self.scoring_words = word_scoring_methods[cf_config.word_scoring_method]
        
    def init_model(self):
        self.model = self.model.to(self.device)
        self._init_cls_module()
        self.model.eval()
        
    def _init_cls_module(self):
        self.model.cls_module.return_cls_embedding = False
        if self.position_scoring_method == 'gradient' or self.word_scoring_method == 'gradient':
            self.model.cls_module.return_bert_embedding = True
         
    def init_input(self, cf_input: CounterfactualInput):
        cf_input = cf_input.to(self.device)
        cf_input.tile_attention_masks(self.number_top_positions, self.max_input_len)
        return cf_input
        
    def _get_word_embedding(self) -> Tensor:
        return self.model.cls_module.bare_bert.embeddings.word_embeddings.weight.unsqueeze(dim=0)
    
    def exp_pred(self, inputs:CounterfactualInput)->Tensor:
        with torch.no_grad():
            mtl_preds = self.mtl_module(inputs=inputs.expred_inputs,
                                        attention_masks=inputs.attention_masks)
        hard_exp_preds = torch.round(mtl_preds['exp_preds'])
        return hard_exp_preds
    
    def cls_pred(self, masked_inputs:CounterfactualInput):
        with torch.no_grad():
            cls_preds = self.cls_module(inputs=masked_inputs.masked_inputs, 
                                        attention_masks=masked_inputs.attention_masks)
        return cls_preds['cls_preds'], cls_preds['bert_embeddings']
           
    def _postprocess(self, cls_preds:Tensor, cf_input:CounterfactualInput):
        pred = int(torch.argmax(cls_preds.data, dim=-1).cpu()[0])
        pred = cf_input.class_name[pred]   
        
        return {
            'input': cf_input.expred_inputs[0],
            'pred': pred,
            'label': cf_input.orig_labels[0]
        }
    
    @classmethod
    def _pred_is_flipped(cls, cf_res:List[Dict[str, Union[Tensor, str]]]) -> bool:
        return cf_res[-1]['pred'] != cf_res[0]['pred']
    
    def _select_candidate_positions(self, position_scores, number_top_positions):
        return torch.argsort(position_scores, dim=1)[:number_top_positions]  # B*top_pos, B=1, L for length of input, top_pos for number of top positions
    
    def _select_candidate_words(self, word_scores, candidate_positions):
        candidate_words = torch.argmax(word_scores, dim=-1)  # B*L
        candidate_words = torch.gather(candidate_words, dim=1, index=candidate_positions)  # B*top_pos
        return candidate_words
    
    def _build_candidate_masked_inputs(self, cf_input, candidate_positions, candidate_words):
        actual_input_length = cf_input.masked_inputs.shape[-1]
        number_top_positions = candidate_positions.shape[-1]
        
        tiled_masked_inputs = torch.tile(torch.unsqueeze(cf_input.masked_inputs, dim=1),
                                         [1, number_top_positions, 1])  # B*top_pos*L, 
        candidate_masked_inputs = torch.scatter(tiled_masked_inputs, dim=-1,
                                                index=candidate_positions.unsqueeze(-1),
                                                src=candidate_words.unsqueeze(-1))  # B*top_pos*L
        candidate_masked_inputs = candidate_masked_inputs.reshape((-1, actual_input_length))
        return candidate_masked_inputs
    
    def _finalize_position_and_word(self,
                                    candidate_positions, candidate_words,
                                    candidate_masked_inputs, tiled_orig_preds, cf_input):
        number_top_positions = candidate_masked_inputs.shape[1]
        with torch.no_grad():
            candidate_cls_preds, _ = self.cls_pred(inputs=candidate_masked_inputs,
                                                   attention_masks=cf_input.tiled_attention_masks)
        candidate_losses = self.loss_criterion(candidate_cls_preds, tiled_orig_preds).reshape((-1, number_top_positions))  # B*top_pos
        positions_rank = torch.argmax(candidate_losses, dim=-1).unsqueeze(-1)  # B*1
        position_to_replace = torch.gather(candidate_positions, dim=1, index=positions_rank)  # B*1
        word_to_replace = torch.gather(candidate_words, dim=1,
                                       index=positions_rank)  # B*1
        return position_to_replace, word_to_replace
        
    def _beam_search(self,
                     position_scores:Tensor, word_scores:Tensor,
                     tiled_orig_preds:Tensor,
                     cf_input:CounterfactualInput):
        candidate_positions = self._select_candidate_positions(position_scores)
        candidate_words = self._select_candidate_words(word_scores, candidate_positions)
        candidate_masked_inputs = self._build_candidate_masked_inputs(cf_input, candidate_positions, candidate_words)
        position_to_replace, word_to_replace = self._finalize_position_and_word(candidate_positions, candidate_words,
                                                                                candidate_masked_inputs, tiled_orig_preds)
        
        return position_to_replace, word_to_replace
            
    def geneate_counterfactuals(self, cf_input: CounterfactualInput):
        cf_input = self.init_input(cf_input)
        if cf_input.custom_input_mask is None:
            rationale_masks = self.exp_pred(cf_input)
        else:
            rationale_masks = cf_input.custom_input_masks
        cf_input.apply_masks_to_inputs(rationale_masks)
        
        current_cls_preds, current_bert_embeddings = self.cls_pred(cf_input.masked_inputs)
        
        orig_cls_preds = torch.argmax(current_cls_preds.data, dim=-1).to(self.device)
        tiled_orig_preds = torch.tile(orig_cls_preds.squeeze(-1),
                                      (1, self.number_top_positions)).reshape((-1)).to(self.device)
        
        cf_res = [self._postprocess(current_cls_preds, cf_input)]
        
        for count_words_replaced in range(self.max_count_word_replacement):
            loss = self.loss_criterion(current_cls_preds, orig_cls_preds).mean()
            grad_wrt_bert_embedding = torch.autograd.grad(loss, current_bert_embeddings)[0].data
                
            position_scores = self.scoring_positions(grad_wrt_bert_embedding, current_bert_embeddings, rationale_masks)
            
            position_scores = position_scores[:, :self.number_top_positions]  # B*top_pos
            
            word_scores = self.scoring_words(grad_wrt_bert_embedding, current_bert_embeddings, self.word_embedding)
            
            position_to_replace, word_to_replace = self._beam_search(position_scores, word_scores,
                                                                     tiled_orig_preds,
                                                                     cf_input)
            
            cf_input.expred_inputs = torch.scatter(cf_input.expred_inputs, dim=1,
                                                   index=position_to_replace,
                                                   src=word_to_replace)
            current_cls_preds, current_bert_embeddings = self.cls_pred(cf_input.masked_inputs)
            cf_res.append(self._postprocess(current_cls_preds, cf_input))
            if count_words_replaced == self.max_count_word_replacement or self._pred_is_flipped(cf_res):
                break
            
        return cf_res
    



