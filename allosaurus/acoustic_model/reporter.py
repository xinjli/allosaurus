from allosaurus.model import get_model_path

class Reporter:

    def __init__(self, train_config):
        self.train_config = train_config

        self.model_path = get_model_path(train_config.new_model)

        # whether write into std
        self.verbose = train_config.verbose

        # log file
        self.log_file = None

        self.open()

    def open(self):
        # whether write into log file
        self.log_file = None
        if self.train_config.log != 'none':
            self.log_file = open(self.model_path / 'log.txt', 'w', encoding='utf-8')

    def close(self):
        if self.log_file:
            self.log_file.close()

    def write(self, message):

        if self.verbose:
            print(message)

        if self.log_file:
            self.log_file.write(message+'\n')