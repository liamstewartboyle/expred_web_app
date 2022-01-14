from expred.expred.config import ExpredConfig

application = 'counterfactual'


class FaxplainConfig(ExpredConfig):
    def __init__(self) -> None:
        super().__init__(dataset_name='fever',
                         mtl_url='https://www.dropbox.com/s/qwinyap4kbxzdvn/evidence_token_identifier.pt?dl=1',
                         cls_url='https://www.dropbox.com/s/oc3qrgl0tqn9mqd/evidence_classifier.pt?dl=1',
                         class_names=["SUPPORTS", "REFUTES"])
        self.load_from_pretrained = True


class CounterfactualConfig(ExpredConfig):
    top_docs = 1
    max_sentence = 30
    max_count_word_replacement = 5
    number_top_positions = 10

    bert_dir = 'bert-base-uncased'

    position_scoring_method = 'gradient'
    word_scoring_method = 'gradient'
    use_custom_mask = False
    masking_method = 'expred'
    constraints = {
        'gramma': False,
    }

    def __init__(self) -> None:
        super().__init__(dataset_name='movies',
                         mtl_url='https://www.dropbox.com/s/qen0vx2uz6ksn3m/evidence_token_identifier.pt?dl=1',
                         cls_url='https://www.dropbox.com/s/0sfrdykcg6cf6kh/evidence_classifier.pt?dl=1',
                         class_names=['NEG', 'POS'])
        self.load_from_pretrained = True

    def update_config_from_ajax_request(self, request) -> None:
        self.position_scoring_method = request.json['position_scoring_method']
        self.word_scoring_method = request.json['word_scoring_method']
        self.use_custom_mask = request.json['use_custom_mask']
        self.masking_method = request.json['masking_method']
        self.constraints = {
            'gramma': request.json['gramma'],
        }
