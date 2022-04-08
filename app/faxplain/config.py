from expred import ExpredConfig

application = 'counterfactual'


class FaxplainConfig(ExpredConfig):
    def __init__(self) -> None:
        super().__init__(dataset_name='movies')
        self.load_from_pretrained = True


class CounterfactualConfig(ExpredConfig):
    top_docs = 1
    max_sentence = 30
    max_count_word_replacement = 5
    number_top_positions = 10

    bert_dir = 'bert-base-uncased'

    selection_strategy = 'hotflip'
    use_custom_mask = False
    masking_method = 'expred'
    constraints = {
        'gramma': False,
    }

    def __init__(self) -> None:
        super().__init__(dataset_name='movies')
        self.load_from_pretrained = True

    def update_config_from_ajax_request(self, request) -> None:
        self.selection_strategy = request.json['selection_strategy']
        self.use_custom_mask = request.json['use_custom_mask']
        self.masking_method = request.json['masking_method']
        self.constraints = {
            'gramma': request.json['gramma'],
        }
