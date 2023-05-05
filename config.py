import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.lr = 3e-4

    config.hidden_dims = (256, 256)

    return config
