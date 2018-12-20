from keras import optimizers


def clone_optimizer(optimizer):
    """ Helper function to clone a given optimizer

    :param optimizer: prototype optimizer
    :return:
    """
    if type(optimizer) is str:
        return optimizers.get(optimizer)

    params = dict([(k, v) for k, v in optimizer.get_config().items()])
    config = {
        'class_name': optimizer.__class__.__name__,
        'config': params,
    }
    new_optimizer = optimizers.deserialize(config)
    return new_optimizer
