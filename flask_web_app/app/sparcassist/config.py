import torch
from expred import ExpredConfig


class CounterfactualConfig(ExpredConfig):
    top_docs = 1
    max_sentence = 30
    max_count_word_replacement = 5
    number_top_positions = 10
    max_input_len = 512

    bert_dir = 'bert-base-uncased'

    selection_strategy = 'hotflip'
    use_custom_mask = False
    masking_method = 'expred'
    constraints = {
        'gramma': False,
    }

    def __init__(self, development=False) -> None:
        device = torch.device('cpu') if development else torch.device('cuda')
        super().__init__(dataset_name='movies', device=device)
        self.load_from_pretrained = True

    def update_config_from_ajax_request(self, request) -> None:
        self.selection_strategy = request.json['selection_strategy']
        self.use_custom_mask = request.json['use_custom_mask']
        self.masking_method = request.json['masking_method']
        self.constraints = {
            'gramma': request.json['gramma'],
        }