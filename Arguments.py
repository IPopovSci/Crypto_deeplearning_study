class Arguments:
    def __init__(self):
        self.args = {
            'ticker': ['^GSPC'],
            'data_types': ['training', 'validation', 'test'],
            'n_components': None,  # n_components is responsible for designating number of features the data will be reduced to
            'time_steps': 1,
            'tokens': [],
            'largest_index': None,
            'training_constant': 0.6,
            'validation_constant': 0.8,
            'batch_size': 128,
            'epochs': 200,
            'output_path': 'data\output',
            'LR': 0.00020000,
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