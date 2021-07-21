class Arguments:
    def __init__(self):
        self.args = {
            'ticker': ['GME'],
            'data_types': ['training', 'validation', 'test'],
            'n_components': None,  # n_components is responsible for designating number of features the data will be reduced to
            'time_steps': 7,
            'tokens': [],
            'largest_index': None,
            'training_constant': 0.6,
            'validation_constant': 0.8,
            'batch_size': 32,
            'epochs': 5,
            'mm_path': None,
            'sc_path': None,
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