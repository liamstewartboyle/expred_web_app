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
