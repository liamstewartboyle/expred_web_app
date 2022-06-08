from expred import ExpredConfig


class FaxplainConfig(ExpredConfig):
    def __init__(self) -> None:
        super().__init__(dataset_name='movies')
        self.load_from_pretrained = True