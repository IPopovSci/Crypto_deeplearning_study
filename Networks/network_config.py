"""Singleton class containing settings for the network configuration."""


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
            'model_type': 'conv2d',
            'epochs': 256,
            'dropout': 0.3,
            'l1_reg': 0.05,
            'l2_reg': 0.05,
            'lr': 0.001}
        self.callbacks = {
            'monitor': 'val_loss',
            'mode': 'min',
            'es_patience': 750,
            'rlr_factor': 0.5,
            'rlr_patience': 15,
        }
