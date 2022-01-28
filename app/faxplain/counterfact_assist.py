from pprint import pprint
from typing import Dict, List, Tuple, Union, Any

import torch
from torch import Tensor, nn
from transformers import BasicTokenizer

from config import CounterfactualConfig
from counterfact_result import CounterfactResults
from expred_utils import Expred
from inputs import CounterfactualInput, ExpredInput
from tokenizer import BertTokenizerWithSpans


def scoring_position_grad(grad_wrt_embd: Tensor, bert_embeddings: Tensor,
                          subtoken_input_rationale_masks: Tensor) -> Tensor:
    position_scores = torch.mean(grad_wrt_embd * bert_embeddings, dim=-1) + (
            1 - subtoken_input_rationale_masks) * 1e16  # B*L
    return position_scores


def scoring_words_grad(grad_wrt_embd: Tensor, word_embedding: Tensor) -> Tensor:
    word_embedding_t = word_embedding.transpose(1, 2)
    word_scores = grad_wrt_embd.matmul(word_embedding_t)
    return word_scores


position_scoring_methods = {'gradient': scoring_position_grad}
word_scoring_methods = {'gradient': scoring_words_grad}


class ExpredCounterAssist:
    def __init__(self, cf_config: CounterfactualConfig, model: Expred):
        # self.cf_config = cf_config
        self.max_number_top_positions = cf_config.number_top_positions
        self.number_top_positions = self.max_number_top_positions
        self.max_count_word_replacement = cf_config.max_count_word_replacement
        self.cf_config = cf_config
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

    def _get_word_embedding(self) -> Tensor:
        return self.model.cls_module.bare_bert.embeddings.word_embeddings.weight.unsqueeze(dim=0)

    def exp_pred(self, inputs: CounterfactualInput) -> Tensor:
        with torch.no_grad():
            mtl_preds = self.model.mtl_module(inputs=inputs.expred_inputs,
                                              attention_masks=inputs.attention_masks)
        hard_exp_preds = torch.round(mtl_preds['exp_preds'])
        rationale_masks = inputs.select_all_overheads(hard_exp_preds)
        return rationale_masks

    def _cls_pred(self, masked_inputs, attention_masks):
        cls_preds = self.model.cls_module(inputs=masked_inputs,
                                          attention_masks=attention_masks)
        return cls_preds['cls_pred'], cls_preds['bert_embeddings']

    @staticmethod
    def _postprocess(cls_preds: Union[Tensor, List[int]],
                     cf_input: ExpredInput,
                     tokenizer: BertTokenizerWithSpans,
                     pos: int,
                     is_subtoken_pos=True):
        if isinstance(cls_preds, Tensor):
            pred = int(torch.argmax(cls_preds.data, dim=-1).cpu()[0])
        else:
            pred = cls_preds[0]
        pred = cf_input.class_names[pred]

        subtoken_counterfactual_doc = cf_input.extract_subtoken_docs(cf_input.expred_inputs)[0]
        token_string_counterfactual_doc = tokenizer.decode(subtoken_counterfactual_doc,
                                                           cf_input.docs_spans[0])

        if pos != -1 and is_subtoken_pos:
            overhead_len = len(cf_input.encoded_queries[0]) + 2
            pos -= overhead_len
            for i, (start, end) in enumerate(cf_input.docs_spans[0]):
                if end > pos:
                    pos = i
                    break

        return {
            'input': token_string_counterfactual_doc,
            'pred': pred,
            'replaced': pos,
            'label': cf_input.orig_labels[0]
        }

    def _select_candidate_positions(self, position_scores):
        return torch.argsort(position_scores, dim=1)[:, :self.number_top_positions]
        # B*top_pos, B=1, L for length of input, top_pos for number of top positions

    @staticmethod
    def _select_candidate_words(word_scores, candidate_positions):
        candidate_words = torch.argmax(word_scores, dim=-1)  # B*L
        candidate_words = torch.gather(candidate_words, dim=1, index=candidate_positions)  # B*top_pos
        return candidate_words

    def _build_candidate_masked_inputs(self, cf_input, candidate_positions, candidate_words, tokenizer=None):
        # actual_input_length = cf_input.masked_inputs.shape[-1]
        number_top_positions = self.number_top_positions

        tiled_masked_inputs = torch.tile(torch.unsqueeze(cf_input.masked_inputs, dim=1),
                                         [1, number_top_positions, 1])  # B*top_pos*L, 

        candidate_masked_inputs = torch.scatter(tiled_masked_inputs, dim=-1,
                                                index=candidate_positions.unsqueeze(-1),
                                                src=candidate_words.unsqueeze(-1))  # B*top_pos*L

        candidate_masked_inputs = candidate_masked_inputs.reshape((number_top_positions, -1))  # top_pos * (B x L)
        return candidate_masked_inputs

    def _finalize_position_and_word(self,
                                    candidate_positions, candidate_words,
                                    candidate_masked_inputs, tiled_orig_preds, cf_input: CounterfactualInput):
        number_top_positions = self.number_top_positions
        with torch.no_grad():
            candidate_cls_preds, _ = self._cls_pred(candidate_masked_inputs, cf_input.tiled_attention_masks)
        candidate_losses = self.loss_criterion(candidate_cls_preds, tiled_orig_preds).reshape(
            (-1, number_top_positions))  # B*top_pos
        positions_rank = torch.argmax(candidate_losses, dim=-1).unsqueeze(-1)  # B*1
        position_to_replace = torch.gather(candidate_positions, dim=1, index=positions_rank)  # B*1
        word_to_replace = torch.gather(candidate_words, dim=1,
                                       index=positions_rank)  # B*1
        return position_to_replace, word_to_replace

    # def compute_subtoken_input_masks(self, cf_input: CounterfactualInput) -> Tuple[Tensor, Tensor]:
    #     if cf_input.subtoken_input_rationale_masks is None:
    #         subtoken_input_rationale_masks = self.exp_pred(cf_input)
    #     else:
    #         subtoken_input_rationale_masks = cf_input.subtoken_input_rationale_masks
    #     subtoken_input_rationale_masks = cf_input.select_all_overheads(subtoken_input_rationale_masks)
    #     subtoken_input_position_masks = cf_input.select_no_overheads(subtoken_input_rationale_masks)
    #     self.number_top_positions = min(int(torch.sum(subtoken_input_position_masks)), self.max_number_top_positions)
    #     return subtoken_input_rationale_masks, subtoken_input_position_masks

    def cls_pred(self, cf_input: ExpredInput) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        current_cls_preds, current_bert_embeddings = self._cls_pred(cf_input.masked_inputs, cf_input.attention_masks)
        argmax_cls_preds = torch.argmax(current_cls_preds.data, dim=-1).to(self.device)
        tiled_orig_preds = torch.tile(argmax_cls_preds.squeeze(-1),
                                      (1, self.number_top_positions)).reshape((-1)).to(self.device)
        return argmax_cls_preds, tiled_orig_preds, current_cls_preds, current_bert_embeddings

    @staticmethod
    def get_result_insufficient_rationale(
            session_id: str,
            cf_input: CounterfactualInput,
            span_tokenizer: BertTokenizerWithSpans) -> CounterfactResults:
        original_instance = ExpredCounterAssist._postprocess(cls_preds=[0],
                                                             cf_input=cf_input,
                                                             tokenizer=span_tokenizer,
                                                             pos=-1)
        ret = CounterfactResults(session_id=session_id,
                                 instances=[original_instance],
                                 mask=cf_input.token_doc_rationale_masks[0],
                                 subtoken_mask=cf_input.subtoken_doc_rationale_masks[0],
                                 ann_id=cf_input.ann_id)
        return ret

    def _do_counterfactual_generation(self,
                                      cf_input: CounterfactualInput,
                                      span_tokenizer: BertTokenizerWithSpans):
        raise NotImplementedError

    @staticmethod
    def _pred_is_flipped(cf_history: List[Dict[str, Union[Tensor, str]]]) -> bool:
        return cf_history[-1]['pred'] != cf_history[0]['pred']

    def geneate_counterfactuals(self,
                                session_id: str,
                                cf_input: CounterfactualInput,
                                span_tokenizer: BertTokenizerWithSpans) -> CounterfactResults:

        # subtoken_input_rationale_masks, subtoken_input_position_masks = self.compute_subtoken_input_masks(cf_input)

        # cf_input.apply_subtoken_input_rationale_masks(subtoken_input_rationale_masks)
        # cf_input.apply_subtoken_input_position_masks(subtoken_input_position_masks)

        if self.number_top_positions == 0:
            return self.get_result_insufficient_rationale(session_id,
                                                          cf_input,
                                                          span_tokenizer)
        else:
            cf_history = self._do_counterfactual_generation(cf_input, span_tokenizer)

        mask = cf_input.token_doc_rationale_masks[0]
        if isinstance(mask, Tensor):
            mask = mask.tolist()
        return CounterfactResults(session_id=session_id,
                                  instances=cf_history,
                                  mask=mask,
                                  subtoken_mask=cf_input.subtoken_doc_rationale_masks[0].tolist(),
                                  ann_id=cf_input.ann_id)


class HotflipCounterAssist(ExpredCounterAssist):
    def _beam_search(self,
                     grad_wrt_bert_embedding: Tensor,
                     current_bert_embeddings: Tensor,
                     tiled_orig_preds: Tensor,
                     cf_input: CounterfactualInput,
                     tokenizer=None):
        position_scores = self.scoring_positions(grad_wrt_bert_embedding,
                                                 current_bert_embeddings,
                                                 cf_input.subtoken_input_position_masks[0])
        position_scores = position_scores  # B*L
        word_scores = self.scoring_words(grad_wrt_bert_embedding, self.word_embedding)

        candidate_positions = self._select_candidate_positions(position_scores)
        candidate_words = self._select_candidate_words(word_scores, candidate_positions)
        candidate_masked_inputs = self._build_candidate_masked_inputs(cf_input, candidate_positions, candidate_words,
                                                                      tokenizer)
        position_to_replace, word_to_replace = self._finalize_position_and_word(candidate_positions, candidate_words,
                                                                                candidate_masked_inputs,
                                                                                tiled_orig_preds,
                                                                                cf_input)
        return position_to_replace, word_to_replace

    def _do_counterfactual_generation(self,
                                      cf_input: CounterfactualInput,
                                      span_tokenizer: BertTokenizerWithSpans):
        (orig_cls_preds,
         tiled_orig_preds,
         current_cls_preds,
         current_bert_embeddings) = self.cls_pred(cf_input)

        cf_input.tile_attention_masks(self.number_top_positions)

        cf_history = [self._postprocess(current_cls_preds, cf_input, span_tokenizer, -1)]
        for count_words_replaced in range(self.max_count_word_replacement):
            loss = self.loss_criterion(current_cls_preds, orig_cls_preds).mean()
            grad_wrt_bert_embedding = torch.autograd.grad(loss, current_bert_embeddings)[0].data

            position_to_replace, word_to_replace = self._beam_search(grad_wrt_bert_embedding,
                                                                     current_bert_embeddings,
                                                                     tiled_orig_preds,
                                                                     cf_input,
                                                                     span_tokenizer)

            cf_input.masked_inputs = torch.scatter(cf_input.masked_inputs, dim=1,
                                                   index=position_to_replace,
                                                   src=word_to_replace)
            cf_input.expred_inputs = torch.scatter(cf_input.expred_inputs, dim=1,
                                                   index=position_to_replace,
                                                   src=word_to_replace)
            current_cls_preds, current_bert_embeddings = self._cls_pred(cf_input.masked_inputs,
                                                                        cf_input.attention_masks)
            counterfactual = self._postprocess(current_cls_preds,
                                               cf_input,
                                               span_tokenizer,
                                               int(position_to_replace.squeeze().data))
            cf_history.append(counterfactual)

            if self._pred_is_flipped(cf_history):
                break

        return cf_history


class MLMCounterAssist(ExpredCounterAssist):
    basic_tokenizer = BasicTokenizer()
    class_full_name = ['negative', 'positive']
    from transformers import pipeline
    unmasker = pipeline('fill-mask')

    @staticmethod
    def get_alternative_preds(preds: Tensor, tiled_preds: Tensor = None) -> Union[Tuple[Tensor, Tensor], Tensor]:
        if tiled_preds is None:
            return 1 - preds
        return 1 - preds, 1 - tiled_preds

    @staticmethod
    def convert_docs(unmasker_results, reference_input):
        doc = reference_input.orig_docs[0]
        ret = []
        for res in unmasker_results:
            pos = res['pos']
            token_str = res['token_str']
            new_doc = doc[: pos] + [token_str] + doc[pos + 1:]
            ret.append([new_doc])
        return ret

    def convert_unmasker_res_to_counterfactual_input(self,
                                                     unmasker_results: List[Dict[str, Any]],
                                                     reference_input: CounterfactualInput,
                                                     span_tokenizer: BertTokenizerWithSpans) -> ExpredInput:
        docs = self.convert_docs(unmasker_results, reference_input)
        # can't use this because of the bug in mlm package,
        # which adds unexpected prefix to some replaced words and causes #token increasing
        # docs = [[self.basic_tokenizer.tokenize(c['sequence'])] for c in unmasker_results]
        queries = reference_input.orig_queries * len(unmasker_results)
        labels = reference_input.orig_labels * len(unmasker_results)
        cf_input = CounterfactualInput(queries=queries,
                                       docs=docs,
                                       labels=labels,
                                       config=self.cf_config,
                                       ann_id=None,
                                       span_tokenizer=span_tokenizer)
        token_doc_rationale_masks = reference_input.token_doc_rationale_masks * len(unmasker_results)
        token_doc_position_masks = reference_input.token_doc_position_masks * len(unmasker_results)
        cf_input.apply_token_doc_rationale_masks(token_doc_rationale_masks)
        cf_input.apply_token_doc_position_masks(token_doc_position_masks)
        return cf_input

    @staticmethod
    def remove_prompt_overhead(unmasker_results):
        for i in range(len(unmasker_results)):
            prompted = unmasker_results[i]['sequence']
            end = len(prompted) - list(reversed(prompted)).index('"') - 1
            cf_doc = prompted[1: end]
            unmasker_results[i]['sequence'] = cf_doc
        return unmasker_results

    def select_best_counterfactual(self, unmasker_results: List[Dict], cf_input, span_tokenizer, alternative_cls_preds):
        # print(unmasker_results[-3])
        # print(unmasker_results[-2])
        expred_input = self.convert_unmasker_res_to_counterfactual_input(unmasker_results,
                                                                         cf_input,
                                                                         span_tokenizer)

        current_cls_preds, _ = self._cls_pred(expred_input.masked_inputs,
                                              expred_input.attention_masks)
        tiled_alternative_cls_preds = torch.tile(alternative_cls_preds, dims=(len(current_cls_preds),))
        losses = self.loss_criterion(current_cls_preds, tiled_alternative_cls_preds)
        selected_id = torch.argmin(losses, dim=-1)

        counterfactual = unmasker_results[selected_id]
        counterfactual['cls_pred'] = current_cls_preds[selected_id].unsqueeze(0)
        return counterfactual

    def get_mlm_counterfactual(self,
                               alternative_cls_preds: Tensor,
                               input_doc: List[str],
                               cf_input: CounterfactualInput,
                               span_tokenizer: BertTokenizerWithSpans):

        query_prompt = f"the sentiment of this review is {self.class_full_name[int(alternative_cls_preds[0])]}."

        mask_token = self.unmasker.tokenizer.mask_token
        unmasker_results = []
        assert len(input_doc) == len(cf_input.token_doc_position_masks[0])
        assert len(input_doc) == len(cf_input.token_doc_rationale_masks[0])
        for pos, (token_str, m) in enumerate(zip(input_doc, cf_input.token_doc_position_masks[0])):
            if token_str == cf_input.wildcard_token or m != 1:
                continue
            doc_first_half = ' '.join(input_doc[:pos])
            doc_second_half = ' '.join(input_doc[pos + 1:])
            prompt_input = f'"{doc_first_half} {mask_token} {doc_second_half}" {query_prompt}'
            for res in self.unmasker(prompt_input):
                if res['token_str'].strip() != token_str:
                    unmasker_results.append(res)
                    unmasker_results[-1]['pos'] = pos
                    break

        unmasker_results = self.remove_prompt_overhead(unmasker_results)
        counterfactual = self.select_best_counterfactual(unmasker_results, cf_input, span_tokenizer,
                                                         alternative_cls_preds)

        return counterfactual

    def _do_counterfactual_generation(self,
                                      cf_input: CounterfactualInput,
                                      span_tokenizer: BertTokenizerWithSpans):
        orig_cls_preds, tiled_orig_preds, current_cls_preds, _ = self.cls_pred(cf_input)

        cf_history = [self._postprocess(current_cls_preds, cf_input, span_tokenizer, -1)]

        alternative_cls_preds = self.get_alternative_preds(orig_cls_preds)

        for count_words_replaced in range(self.max_count_word_replacement):
            # print(cf_input.orig_docs)
            input_doc = cf_input.orig_docs[0]

            counterfactual = self.get_mlm_counterfactual(alternative_cls_preds,
                                                         input_doc,
                                                         cf_input,
                                                         span_tokenizer)

            cf_input = self.convert_unmasker_res_to_counterfactual_input([counterfactual],
                                                                         cf_input,
                                                                         span_tokenizer)
            unmasker_res = self._postprocess(counterfactual['cls_pred'],
                                             cf_input,
                                             span_tokenizer,
                                             counterfactual['pos'],
                                             is_subtoken_pos=False)
            cf_history.append(unmasker_res)

            if self._pred_is_flipped(cf_history):
                break

        return cf_history
