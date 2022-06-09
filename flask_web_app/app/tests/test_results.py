from unittest import TestCase

from sparcassist import CounterfactResults


class TestExpredAssist(TestCase):
    def test_combine_cf_results(self):
        cf_res_0 = {
            'instances': [
                {
                    'input': ['one', 'two', 'three'],
                    'pred': ['POS'],
                    'replaced': -1,
                    'label': ['NEG'],
                }, {
                    'input': ['one', 'two', 'four'],
                    'pred': ['POS'],
                    'replaced': 2,
                    'label': ['NEG'],
                    'alternative_words': ['five', 'seven', 'eight']
                }, {
                    'input': ['one', 'b', 'four'],
                    'pred': ['POS'],
                    'replaced': 1,
                    'label': ['NEG'],
                    'alternative_words': ['a', 'c', 'd']
                }],
            'mask': [1, 1, 1],
            'subtoken_mask': [1, 1, 1],
            'ann_id': 'bogus'
        }

        divide_id = 1

        cf_res_1 = {
            'cf_examples': {
                'instances': [
                    {
                        'input': ['one', 'two', 'five'],
                        'pred': ['POS'],
                        'replaced': -1,
                        'label': ['NEG'],
                    }, {
                        'input': ['1', 'two', 'five'],
                        'pred': ['POS'],
                        'replaced': 2,
                        'label': ['POS'],
                        'alternative_words': ['2', '3', '4']
                    }],
                'mask': [1, 1, 1],
                'subtoken_mask': [1, 1, 1],
                'ann_id': 'bogus'
            },
            'session_id': 1234
        }

        cf_res_should = {
            'cf_examples': {
                'instances': [
                    {
                        'input': ['one', 'two', 'three'],
                        'pred': ['POS'],
                        'replaced': -1,
                        'label': ['NEG'],
                    }, {
                        'input': ['one', 'two', 'five'],
                        'pred': ['POS'],
                        'replaced': -1,
                        'label': ['NEG'],
                    }, {
                        'input': ['1', 'two', 'five'],
                        'pred': ['POS'],
                        'replaced': 2,
                        'label': ['POS'],
                        'alternative_words': ['2', '3', '4']
                    }],
                'mask': [1, 1, 1],
                'subtoken_mask': [1, 1, 1],
                'ann_id': 'bogus'
            },
            'session_id': 1234
        }
        cf_res_1 = CounterfactResults.from_dict(cf_res_1)
        ret = CounterfactResults.combine_cf_results(cf_res_0, divide_id, cf_res_1)
        self.assertDictEqual(ret.to_dict(), cf_res_should)
