from torch.optim import SGD

def read_optimizer(model, train_config):

    assert train_config.optimizer == 'sgd', 'only sgd is supported now, others optimizers would be easier to add though'

    return SGD(model.parameters(), lr=train_config.lr)