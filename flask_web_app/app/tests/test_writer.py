import pickle
import unittest

import re
from copy import deepcopy

from sparcassist.config import CounterfactualConfig
from sparcassist.writer import CounterfactWriter
from sparcassist.inputs import CounterfactualInput
from expred.expred.tokenizer import BertTokenizerWithSpans

span_tokenizer = BertTokenizerWithSpans.from_pretrained('bert-base-uncased')

query = 'what is that ?'.split()
doc = 'that be a bird . and this is a dog .'.split()
label = 'POS'
ann_id = 'test_0'

cf_history = [
    {'replaced': -1,
     'input': 'that be a bird . and this is a dog .'.split(),
     'pred': 'POS'},
    {'replaced': 1,
     'input': 'that is a bird . and this is a dog .'.split(),
     'pred': 'POS'},
    {'replaced': 1,
     'input': 'that was a bird . and this is a dog .'.split(),
     'pred': 'NEG'}
]

history_should = [{'pos': 1,
                   'word': 'is',
                   'pred': 'POS'},
                  {'pos': 1,
                   'word': 'was',
                   'pred': 'NEG'}]

cf_res = {
    'cf_examples': {
        'mask': [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        'subtoken_mask': [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        'instances': cf_history
    },
    'ann_id': ann_id
}

output_should = {'ann_id': ann_id,
                 'sentence': doc,
                 'token_mask': [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                 'subtoken_mask': [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                 'masking_method': 'expred',
                 'position_scoring_method': 'gradient',
                 'word_scoring_method': 'gradient',
                 'gramma': False,
                 'history': history_should}

eval_should = deepcopy(output_should)
eval_should['plausibility'] = 3
eval_should['meaningfulness'] = 4
eval_should['risk'] = 3
config = CounterfactualConfig()
cf_input = CounterfactualInput(queries=[query],
                               docs=[doc],
                               labels=[label],
                               config=config,
                               ann_ids=ann_id,
                               span_tokenizer=span_tokenizer)
cf_input.counterfactual_results = cf_res

writer = CounterfactWriter()

res_fname_should = f'counterfactual_res/res_{writer.session_id}.pkl'

eval_fname_should = f'counterfactual_res/eval_{writer.session_id}.pkl'

eval_mock = {'plausibility': 3,
             'meaningfulness': 4,
             'risk': 5}


class TestCounterfactWriter(unittest.TestCase):
    def test_session_id(self):
        sessionid_pattern = re.compile("[0-9a-f]{8}")
        self.assertTrue(sessionid_pattern.match(writer.session_id))

    def test_get_cf_history(self):
        history = writer.get_counterfactual_history(cf_history[1:])
        self.assertSequenceEqual(history, history_should)

    def test___get_output_data(self):
        output = writer._get_output_data(cf_input, config)
        self.assertDictEqual(output, output_should)

    def test_write_cf_example(self):
        writer.write_cf_example(cf_input, config)
        with open(res_fname_should, 'rb') as fin:
            res = pickle.load(fin)
        self.assertDictEqual(output_should, res)

    def test_write_evaluation(self):
        writer.write_evaluation(cf_input, config, eval_mock)
        with open(eval_fname_should, 'rb') as fin:
            res = pickle.load(fin)
        self.assertSequenceEqual(eval_should, res)
