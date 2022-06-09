import csv
import os
import pickle
from random import random
from typing import Dict, List, Union, Any

from flask import Request

from .config import CounterfactualConfig
from .results import CounterfactResults


class FaxplainWriter:
    base_dir = '../'

    def __init__(self, session_id: str = None, res_folder=None, create_res_file=True) -> None:
        if not session_id:
            self.session_id = hex(int(random() * 1e13))[2:]
        else:
            self.session_id = session_id

        self.res_folder = self.base_dir + 'res' if res_folder is None else res_folder
        FaxplainWriter.maybe_create_res_folder(self.res_folder)

        self.ugc_data_fname = f'{self.res_folder}/ugc_{self.session_id}.csv'  # user generated content
        self.mgc_data_fname = f'{self.res_folder}/mgc_{self.session_id}.csv'  # machine genarated content
        self.temp_data_fname = self.res_folder + f'data/temp_{self.session_id}.pkl'

        if create_res_file:
            FaxplainWriter.maybe_create_res_file(self.ugc_data_fname)
            FaxplainWriter.maybe_create_res_file(self.mgc_data_fname)

    @classmethod
    def maybe_create_res_file(cls, fname):
        if not os.path.isfile(fname):
            with open(fname, 'w+', newline='') as fout:
                writer = csv.writer(fout)
                writer.writerow('query url evidence label'.split())

    @classmethod
    def maybe_create_res_folder(cls, dirname):
        os.makedirs(dirname, exist_ok=True)


class CounterfactWriter(FaxplainWriter):
    def __init__(self, request: Request) -> None:
        if 'session_id' not in request.json:
            request.json['session_id'] = None
        res_folder = self.base_dir + 'counterfactual_res/'
        super().__init__(session_id=request.json['session_id'],
                         res_folder=res_folder,
                         create_res_file=False)

        # both from user-intervened and from machine generated counterfactuals
        self.res_fname = f'{self.res_folder}/res_{self.session_id}.pkl'
        self.eval_fname = f'{self.res_folder}/eval_{self.session_id}.pkl'

    @staticmethod
    def get_counterfactual_history(cf_history: List[Dict[str, Union[Union[List[str], int], Any]]]):
        history = []
        for cf_example in cf_history:
            pos = cf_example['replaced']
            word = cf_example['input'][pos]
            pred = cf_example['pred']
            history.append({'pos': pos,
                            'word': word,
                            'pred': pred})
        return history

    def _get_output_data(self,
                         cf_results: Union[CounterfactResults, Dict],
                         cf_conf: CounterfactualConfig):
        # if isinstance(cf_results, dict):
        #     ann_id = cf_results['ann_id']
        #     session_id = cf_conf.
        ann_id = cf_results.ann_id
        session_id = cf_results.session_id

        masking_method = cf_conf.masking_method
        selection_strategy = cf_conf.selection_strategy
        gramma = cf_conf.constraints['gramma']

        token_mask = cf_results.mask
        subtoken_mask = cf_results.subtoken_mask
        sentence = cf_results.instances[0]['input']

        history = self.get_counterfactual_history(cf_results.instances[1:])
        return {'ann_id': ann_id,
                'session_id': session_id,
                'sentence': sentence,
                'token_mask': token_mask,
                'subtoken_mask': subtoken_mask,
                'masking_method': masking_method,
                'selection_strategy': selection_strategy,
                'gramma': gramma,
                'history': history}

    def write_cf_example(self,
                         cf_results: Union[CounterfactResults, Dict],
                         cf_conf: CounterfactualConfig):
        output = self._get_output_data(cf_results, cf_conf)
        with open(self.res_fname, 'ab+') as fout:
            pickle.dump(output, fout)

    def write_evaluation(self,
                         cf_results: CounterfactResults,
                         cf_conf: CounterfactualConfig,
                         eval: Dict):
        output = self._get_output_data(cf_results, cf_conf)
        output['plausibility'] = eval['plausibility']
        output['meaningfulness'] = eval['meaningfulness']
        output['risk'] = eval['risk']
        with open(self.eval_fname, 'ab+') as fout:
            pickle.dump(output, fout)
