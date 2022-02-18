class Arguments:
    def __init__(self):
        self.args = {
            'starting_date': "1991-01-02",
            'ticker': 'bnbusdt', #ethusd,bnbusdt
            'train_size': 0.85,
            'test_size': 0.15,
            'target_features':None,
            'data_types': ['training', 'validation', 'test'],
            'n_components': None,  # n_components is responsible for designating number of features the data will be reduced to
            'time_steps': 25,
            'batch_size': 128,
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