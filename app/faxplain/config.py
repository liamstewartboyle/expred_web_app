import os
from torch import device

application = 'counterfactual'

class LoadableModelConfig():
    def __init__(self, dataset_name)->None:
        self.dataset_name = dataset_name
        
        
class ExpredConfig(LoadableModelConfig):
    def __init__(self, dataset_name) -> None:
        super().__init__(dataset_name)
        self.dataset_base_dir = os.environ.get('HOME') + '/.keras/datasets/'
        self.device = device('cpu')
        self.class_name = None
        self.wildcard_token = '.'
        self.max_input_len = 512
        
        
class FaxplainConfig(ExpredConfig):
    def __init__(self, dataset_name) -> None:
        super().__init__(dataset_name)
        
        self.debug = False
        
        self.device = device('cpu')
        
        self.mtl_loc = './trained_models/fever/evidence_token_identifier.pt'
        self.mtl_url = 'https://www.dropbox.com/s/qwinyap4kbxzdvn/evidence_token_identifier.pt?dl=1'
        self.cls_loc = 'trained_models/fever/evidence_classifier.pt'
        self.cls_url = 'https://www.dropbox.com/s/oc3qrgl0tqn9mqd/evidence_classifier.pt?dl=1'
        self.class_names = ["SUPPORTS", "REFUTES"]
        
        
class CountefactualConfig(ExpredConfig):
    def __init__(self) -> None:
        super().__init__('movies')
        
        self.evi_finder_url = 'https://www.dropbox.com/s/qen0vx2uz6ksn3m/evidence_token_identifier.pt?dl=1'
        self.cls_url = 'https://www.dropbox.com/s/0sfrdykcg6cf6kh/evidence_classifier.pt?dl=1'
        
        self.class_name = ['NEG', 'POS']
        
        self.top_docs = 1
        self.max_sentence = 30
        self.max_count_word_replacement = 5
        self.number_top_positions = 10
        
        self.position_scoring_method = 'gradient'
        self.word_scoring_method = 'gradient'
        self.constraints = {
            'gramma': False,
        }
        
        self.device = device('cpu')

    def update_config_from_ajax_request(self, request) -> None:
        self.position_scoring_method = request.json['position_scoring_method']
        self.word_scoring_method = request.json['word_scoring_method']
        self.constraints = {
            'gramma': request.json['gramma'],
        }

if application == 'counterfactual':
    cf_config = CountefactualConfig()