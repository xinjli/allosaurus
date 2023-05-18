from torch.optim import Adadelta, Adam, SGD, AdamW

def read_optimizer(model, config):
    if "optimizer" in config:
        if config.optimizer == 'adadelta':
            print("using adadelta lr: " + str(config.lr_rate))
            optimizer = Adadelta(model.model.parameters(), lr=config.lr_rate, eps=1e-8, weight_decay=0.0)
        elif config.optimizer == 'adam':
            print("using adam lr: " + str(config.lr_rate))
            optimizer = Adam(model.model.parameters(), lr=config.lr_rate)
        elif config.optimizer == 'adamw':
            print("using adamW lr: " + str(config.lr_rate))
            optimizer = AdamW(model.model.parameters(), lr=config.lr_rate)
        else:
            print("using sgd lr:" + str(config.lr_rate))
            optimizer = SGD(model.model.parameters(), lr=config.lr_rate)
    else:
        print("using default sgd lr:" + str(config.lr_rate))
        optimizer = SGD(model.model.parameters(), lr=config.lr_rate)

    return optimizer
