class Arguments:
    def __init__(self):
        self.args = {
            'ticker': '^IXIC',
            'data_types': ['training', 'validation', 'test'],
            'n_components': None,  # n_components is responsible for designating number of features the data will be reduced to
            'time_steps': 1,
            'tokens': [],
            'largest_index': None,
            'training_constant': 0.6,
            'validation_constant': 0.8,
            'batch_size': 32,
            'epochs': 256,
            'mm_path': None,
            'sc_path': None,
            'output_path': 'data\output',
            'LR': 0.00000010000,
            'split_constants': {
                'training': None,
                'validation': None,
                'test': None
            },
            'split_index': {
                'training': None,
                'validation': None,
                'test': None
            },
            'parent': 'C:/Users/Ivan/PycharmProjects/MlFinancialAnal/data/scalers/'

        }


args = Arguments().args