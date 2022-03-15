class NetworkParams:
    __instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if NetworkParams.__instance == None:
            NetworkParams()
        return NetworkParams.__instance
    def __init__(self):
        """ Virtually private constructor. """
        if NetworkParams.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            NetworkParams.__instance = self

        self.data = {
        }
        self.network = {
            'model_type': 'dense',
            'epochs': 256,
            'dropout': 0.2,
            'l1_reg': 0.0005,
            'l2_reg': 0.0005,
            'lr': 0.00001}
        self.callbacks = {
            'monitor': 'val_loss',
            'mode': 'min',
            'es_patience': 125,
            'rlr_factor': 0.5,
            'rlr_patience': 5,
        }