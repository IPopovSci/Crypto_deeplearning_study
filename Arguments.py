class Arguments:
    def __init__(self):
        self.args = {
            'ticker': ['^GSPC'],
            'data_types': ['training', 'validation', 'test'],
            'n_components': 40,  # n_components is responsible for designating number of features the data will be reduced to
            'time_steps': 30,
            'tokens': [],
            'largest_index': None,
            'training_constant': 0.6,
            'validation_constant': 0.8,
            'batch_size': 50,
            'epochs': 100,
            'output_path': 'data\output',
            'LR': 0.00010000,
            'split_constants': {
                'training': None,
                'validation': None,
                'test': None
            },
            'split_index': {
                'training': None,
                'validation': None,
                'test': None
            }

        }


args = Arguments().args