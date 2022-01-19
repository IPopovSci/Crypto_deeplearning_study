class Arguments:
    def __init__(self):
        self.args = {
            'starting_date': "1991-01-02",
            'ticker': 'ethusd',
            'train_size': 0.99,
            'test_size': 0.01,
            'target_features':None,
            'data_types': ['training', 'validation', 'test'],
            'n_components': None,  # n_components is responsible for designating number of features the data will be reduced to
            'time_steps': 20,
            'batch_size': 256,
            'epochs': 256,
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