class PipelineArgs:
    __instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if PipelineArgs.__instance == None:
            PipelineArgs()
        return PipelineArgs.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if PipelineArgs.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            PipelineArgs.__instance = self

        self.args = {
            'ticker': 'btcusd', #ethusd,bnbusdt
            'mode': 'training',
            'interval': '1h',
            'train_size': 0.95,
            'test_size': 0.05,
            'time_steps': 15,
            'batch_size': 128,
            'data_lag': 1,
            'ta': True,
            'pca': True,
            'initial_training': True,
            'expand_dims': False, #This setting will add a dimension of 1 at the end of the data, required for some models
            'mm_y_path': None,
            'ss_y_path': None,
        }

