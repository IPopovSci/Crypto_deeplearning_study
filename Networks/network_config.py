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
            'epochs': 256,
            'LR': 0.00000010000,
            'dropout': 0.3,
            'l1_reg': 0.0005,
            'l2_reg': 0.0005,
            'lr': 0.0001

        }