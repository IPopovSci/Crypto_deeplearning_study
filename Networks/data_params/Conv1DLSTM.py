from Arguments import args

class Conv1DLSTM_Network:
    def __init__(self):
        self.data = {
            'ticker': args['ticker'], #ethusd,bnbusdt
            'train_size': 0.85,
            'test_size': 0.15,
            'target_features':None,
            'n_components': None,  # n_components is responsible for designating number of features the data will be reduced to
            'data_from': 'CSV',
            'ta': True,
            'initial_training': True,
            'batch': False,
            'y_type': "testing",
            'pca': False,
        }
        self.network = {
            'epochs': 256,
            'LR': 0.00000010000,

        }
        self.globals = {
            'time_steps': 25,
            'batch_size': 128,
        }