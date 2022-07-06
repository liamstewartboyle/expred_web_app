import torch
from expred import ExpredConfig


class CounterfactualConfig(ExpredConfig):
    top_docs = 1
    max_sentence = 30
    max_count_word_replacement = 5
    number_top_positions = 10
    number_top_candidate_words = 5
    max_input_len = 512

    bert_dir = 'bert-base-uncased'

    selection_strategy = 'hotflip'
    use_custom_mask = False
    masking_method = 'expred'
    constraints = {
        'gramma': False,
    }

    def __init__(self, development=False) -> None:
        if development or not torch.cuda.is_available():
            device = 'cpu'
        else:
            device = torch.device('cuda')
        super().__init__(dataset_name='movies', device=device)
        if torch.cuda.is_available():
            assert self.device == 'cuda' or self.device == torch.device('cuda')
        else:
            assert self.device == 'cpu'
        self.load_from_pretrained = True

    def update_config_from_ajax_request(self, request) -> None:
        self.selection_strategy = request.json['selection_strategy']
        self.use_custom_mask = request.json['use_custom_mask']
        self.masking_method = request.json['masking_method']
        self.constraints = {
            'gramma': request.json['gramma'],
        }
