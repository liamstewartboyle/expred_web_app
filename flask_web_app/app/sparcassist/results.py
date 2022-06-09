from typing import List, Dict, Any, Union


class CounterfactResults:
    def __init__(self,
                 session_id: str,
                 instances: List[Dict[str, Union[Union[List[str], int], Any]]],
                 mask: List[int],
                 subtoken_mask: List,
                 ann_id: str
                 ):
        self.session_id = session_id
        self.instances = instances
        self.mask = mask
        self.subtoken_mask = subtoken_mask
        self.ann_id = ann_id

    def to_dict(self):
        return {
            'cf_examples': {
                'instances': self.instances,
                'mask': self.mask,
                'subtoken_mask': self.subtoken_mask,
                'ann_id': self.ann_id
            },
            'session_id': self.session_id,

        }

    def __repr__(self):
        return self.to_dict().__repr__()

    @classmethod
    def combine_cf_results(cls, cf_res_0: Dict, divide_pos: int, cf_res_1):
        combined_res = {
            'instances': cf_res_0['instances'][:divide_pos],
            'mask': cf_res_0['mask'],
            'subtoken_mask': cf_res_0['mask'],
            'ann_id': cf_res_0['ann_id']
        }
        cf_res_1 = cf_res_1.to_dict()
        combined_res['instances'] += cf_res_1['cf_examples']['instances']
        combined_res['instances'][divide_pos]['replaced'] = cf_res_0['instances'][divide_pos]['replaced']
        ret = {
            'cf_examples': combined_res,
            'session_id': cf_res_1['session_id']
        }
        return cls.from_dict(ret)

    @classmethod
    def from_dict(cls, input_dict: Dict):
        instances = input_dict['cf_examples']['instances']
        mask = input_dict['cf_examples']['mask']
        ann_id = input_dict['cf_examples']['ann_id']
        subtoken_mask = input_dict['cf_examples']['subtoken_mask']
        session_id = input_dict['session_id']
        return cls(session_id,
                   instances,
                   mask,
                   subtoken_mask,
                   ann_id)
